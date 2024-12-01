import asyncio
import json
import threading
import uvloop
import uvicorn
from http import HTTPStatus
import multiprocessing as mp
from dataclasses import asdict
from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scratchpad.managers import TokenizerManager, run_detokenizer_process
from scratchpad.scheduler.scheduler import run_scheduler_process
from scratchpad.server.metrics import PrometheusStatLogger
from scratchpad.server.openai_api.handler import (
    load_chat_template_for_openai_api,
    v1_batches,
    v1_cancel_batch,
    v1_chat_completions,
    v1_completions,
    v1_delete_file,
    v1_embeddings,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from scratchpad.server.controller import (
    mount_metrics,
    start_controller,
    mount_controller,
)
from scratchpad.server.middlewares import add_api_key_middleware
from scratchpad.server.openai_api.protocol import ModelCard, ModelList

from .args import ServerArgs
from .protocol import GenerateReqInput
from scratchpad.server.controller import get_controller

setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

app = FastAPI()
mount_metrics(app)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
tokenizer_manager = None
server_args = None


@app.get("/system_info")
async def system_info():
    return JSONResponse(status_code=200, content={"system_info": asdict(server_args)})


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


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, raw_request)


@app.post("/v1/embeddings")
async def openai_v1_embeddings(raw_request: Request):
    response = await v1_embeddings(tokenizer_manager, raw_request)
    return response


@app.get("/v1/models")
def available_models():
    """Show available models."""
    served_model_names = [tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(ModelCard(id=served_model_name, root=None))
    for extra_model in get_controller().get_toppings():
        model_cards.append(ModelCard(id=extra_model, root=served_model_name))
    return ModelList(data=model_cards)


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    return await v1_files_create(
        file, purpose, tokenizer_manager.server_args.file_storage_pth
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    return await v1_delete_file(file_id)


@app.post("/v1/batches")
async def openai_v1_batches(raw_request: Request):
    return await v1_batches(tokenizer_manager, raw_request)


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batches(batch_id: str):
    # https://platform.openai.com/docs/api-reference/batch/cancel
    return await v1_cancel_batch(tokenizer_manager, batch_id)


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return await v1_retrieve_batch(batch_id)


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve
    return await v1_retrieve_file(file_id)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve-contents
    return await v1_retrieve_file_content(file_id)


def launch_server(model_name, args: "ServerArgs"):
    global tokenizer_manager
    global server_args
    global controller

    server_args = args
    args.model_path = model_name
    args.translate_auto()
    if args.api_key:
        add_api_key_middleware(app, args.api_key)

    # loggers = [PrometheusStatLogger(1, {"server_id": args.server_id}, 4096)]
    # Launch tensor parallel scheduler processes
    scheduler_procs = []
    scheduler_pipe_readers = []
    tp_size_per_node = args.tp_size // args.nnodes
    tp_rank_range = range(
        tp_size_per_node * args.node_rank,
        tp_size_per_node * (args.node_rank + 1),
    )

    for tp_rank in tp_rank_range:
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = tp_rank % tp_size_per_node
        proc = mp.Process(
            target=run_scheduler_process,
            args=(args, gpu_id, tp_rank, writer, []),
        )
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    if args.node_rank >= 1:
        # For other nodes, they do not need to run tokenizer or detokenizer,
        # so they can just wait here.
        while True:
            pass
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(args,),
    )
    detoken_proc.start()
    start_controller(args)
    mount_controller(app)

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)
    for i in range(len(scheduler_pipe_readers)):
        scheduler_pipe_readers[i].recv()
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5, loop="auto")
