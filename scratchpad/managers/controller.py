import zmq
import zmq.asyncio
from scratchpad.server.args import ServerArgs
from .structs import MemoryPoolControlReqInput


class SystemController:
    """SystemController is a process that controls the internal of Scratchpad"""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = context.socket(zmq.PUSH)
        self.send_to_scheduler.connect(f"tcp://127.0.0.1:{server_args.scheduler_port}")

    def control_memory_pool(self, input: MemoryPoolControlReqInput):
        self.send_to_scheduler.send_pyobj(input)
        return True
