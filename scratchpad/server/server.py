import asyncio
import json
import multiprocessing as mp
import os
import threading
import aiohttp
import requests
import uvloop
import uvicorn
from fastapi import FastAPI, Request
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scratchpad.utils import logger
from scratchpad.utils.hf import download_from_hf
from scratchpad.managers import TokenizerManager, start_detokenizer_process
from .args import ServerArgs
from .protocol import GenerateReqInput
from scratchpad.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)

setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

app = FastAPI()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
tokenizer_manager = None


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200, content="OK")


async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results():
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await tokenizer_manager.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


def launch_server(model_name, args: "ServerArgs"):
    global tokenizer_manager
    args.model_path = model_name
    args.translate_auto()
    pipe_controller_reader, pipe_controller_writer = mp.Pipe(duplex=False)

    start_controller_process = start_controller_process_single
    proc_controller = mp.Process(
        target=start_controller_process,
        args=(args, pipe_controller_writer),
    )
    proc_controller.start()
    tokenizer_manager = TokenizerManager(args)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    controller_init_state = pipe_controller_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if controller_init_state != "init ok" or detoken_init_state != "init ok":
        proc_controller.kill()
        proc_detoken.kill()
        raise RuntimeError(
            "Initialization failed. "
            f"controller_init_state: {controller_init_state}, "
            f"detoken_init_state: {detoken_init_state}"
        )
    assert proc_controller.is_alive() and proc_detoken.is_alive()

    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5, loop="auto")
