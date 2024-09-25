import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import threading
import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, Request
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scratchpad.utils import logger
from scratchpad.utils.hf import download_from_hf
from .protocol import GenerateReqInput

setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

app = FastAPI()

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

def launch_server(model_name, args):
    download_from_hf(model_name)
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=5,
            loop="auto"
        )
    except Exception as e:
        logger.error(f"Error in server: {e}")
        os._exit(1)
