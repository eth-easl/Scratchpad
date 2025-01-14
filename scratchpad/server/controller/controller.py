import os
import re
from fastapi import FastAPI
import multiprocessing as mp
from starlette.routing import Mount, Route
from scratchpad.managers.controller import SystemController
from scratchpad.managers.structs import (
    MemoryPoolControlReqInput,
    RegisterToppingsReqInput,
)
from scratchpad.utils import logger
from fastapi.responses import JSONResponse
from typing import TYPE_CHECKING
from scratchpad.utils.toppings.topping_utils import parse_topping_config

if TYPE_CHECKING:
    from scratchpad.server.args import ServerArgs

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


async def register_toppings(request: "RegisterToppingsReqInput"):
    # (todo:xiaozhe): For some reason, the request is not being parsed correctly
    #  so we parse it manually now.
    req = await request.json()
    controller.add_topping(req["model_path_or_name"], req["toppings_type"])
    return JSONResponse(content={"message": "Toppings registered"})


def mount_controller(app: FastAPI):
    app.routes.append(
        Route("/memory_pool/increase", increase_memory_pool_size, methods=["GET"])
    )
    app.routes.append(Route("/toppings", register_toppings, methods=["POST"]))


def start_controller(args: "ServerArgs"):
    global controller
    controller = SystemController(args)
    controller.states.root_model = args.model_path
    if args.init_toppings:
        toppings = parse_topping_config(args.init_toppings)
        for topping in toppings:
            controller.add_topping(
                RegisterToppingsReqInput(
                    topping_type=topping[0],
                    model_path_or_name=topping[1],
                    served_name=topping[2],
                )
            )


def get_controller():
    return controller


def get_toppings():
    return controller.get_toppings()


def model_to_topping(model_name):
    return controller.states.get_topping_path(model_name)
