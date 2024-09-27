import logging
import multiprocessing
from typing import List

import zmq

from .tp_worker import (
    ModelTpServer,
    broadcast_recv_input,
    launch_tp_servers,
)
from scratchpad.server.args import ServerArgs
from scratchpad.utils import get_exception_traceback, kill_parent_process

logger = logging.getLogger(__name__)


class ControllerSingle:
    """A controller that manages a group of tensor parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_ids: List[int],
        is_data_parallel_worker: bool,
        dp_worker_id: int,
        mp_queue: multiprocessing.Queue,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.is_dp_worker = is_data_parallel_worker
        self.dp_worker_id = dp_worker_id
        self.mp_queue = mp_queue

        # Init inter-process communication
        context = zmq.Context(2)

        if not self.is_dp_worker:
            self.recv_from_tokenizer = context.socket(zmq.PULL)
            self.recv_from_tokenizer.bind(
                f"tcp://127.0.0.1:{server_args.controller_port}"
            )

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{server_args.detokenizer_port}"
        )

        # Launch other tp ranks
        tp_size_local = server_args.tp_size // server_args.nnodes
        self.tp_procs = []
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                server_args.nccl_ports[dp_worker_id],
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            server_args.nccl_ports[dp_worker_id],
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def loop_for_forward(self):
        while True:
            if not self.is_dp_worker:
                recv_reqs = self.recv_requests_from_zmq()
            else:
                recv_reqs = self.recv_requests_from_mp_queue()

            if self.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

    def recv_requests_from_zmq(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        return recv_reqs

    def recv_requests_from_mp_queue(self):
        recv_reqs = []
        while not self.mp_queue.empty():
            recv_reqs.append(self.mp_queue.get())
        return recv_reqs


def start_controller_process(
    server_args: ServerArgs,
    pipe_writer: multiprocessing.connection.Connection,
    is_data_parallel_worker: bool = False,
    gpu_ids: List[int] = None,
    dp_worker_id: int = None,
    queue: multiprocessing.connection.Connection = None,
):
    """Start a controller process."""
    # if is_data_parallel_worker:
    #     logger_prefix = f" DP{dp_worker_id} TP0"
    # else:
    #     logger_prefix = " TP0"
    # configure_logger(server_args, prefix=logger_prefix)

    if not is_data_parallel_worker:
        tp_size_local = server_args.tp_size // server_args.nnodes
        gpu_ids = [i for _ in range(server_args.nnodes) for i in range(tp_size_local)]
        dp_worker_id = 0
        queue = None

    try:
        controller = ControllerSingle(
            server_args,
            gpu_ids,
            is_data_parallel_worker,
            dp_worker_id,
            queue,
        )
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        kill_parent_process()
