import os
import re
from fastapi import FastAPI
import multiprocessing as mp
from starlette.routing import Mount, Route
from scratchpad.managers.controller import SystemController
from scratchpad.managers.structs import MemoryPoolControlReqInput
from scratchpad.utils import logger
from fastapi.responses import JSONResponse

from .args import ServerArgs

controller: SystemController = None

"""Prometheus metrics"""


def mount_metrics(app: FastAPI):
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.info(
            f"Scratchpad to use {prometheus_multiproc_dir_path} as PROMETHEUS_MULTIPROC_DIR"
        )
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


"""Controller routes"""


async def increase_memory_pool_size(request):
    controller.control_memory_pool(
        MemoryPoolControlReqInput(is_expand=True, delta=1000)
    )
    return JSONResponse(content={"message": "Memory pool size increased by 50"})


def mount_controller(app: FastAPI):
    app.routes.append(
        Route("/memory_pool/increase", increase_memory_pool_size, methods=["GET"])
    )


def start_controller(args: ServerArgs):
    global controller
    controller = SystemController(args)
