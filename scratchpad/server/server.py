import asyncio
import json
import multiprocessing as mp
import threading
import uvloop
import uvicorn
from fastapi import FastAPI, Request, File, Form, UploadFile
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scratchpad.managers import TokenizerManager, start_detokenizer_process
from .args import ServerArgs
from .protocol import GenerateReqInput
from scratchpad.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
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
from scratchpad.server.openai_api.protocol import ModelCard, ModelList

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
        model_cards.append(ModelCard(id=served_model_name, root=served_model_name))
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
