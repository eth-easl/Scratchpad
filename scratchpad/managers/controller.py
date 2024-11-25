import zmq
import zmq.asyncio
from scratchpad.server.args import ServerArgs
from .structs import MemoryPoolControlReqInput
from .engine_state import EngineStates


class SystemController:
    """SystemController is a process that controls the internal of Scratchpad"""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        context = zmq.asyncio.Context(2)
        self.send_to_scheduler = context.socket(zmq.PUSH)
        self.send_to_scheduler.connect(f"tcp://127.0.0.1:{server_args.scheduler_port}")
        self.states = EngineStates()

    def control_memory_pool(self, input: MemoryPoolControlReqInput):
        self.send_to_scheduler.send_pyobj(input)
        return True

    def add_topping(self, topping_name: str, topping_type: str) -> bool:
        # check if registration is successful
        self.states.add_toppings(topping_name, topping_type)
        return True

    def get_toppings(self):
        return self.states.toppings
