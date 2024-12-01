from scratchpad.server.args import ServerArgs
from scratchpad.utils import logger
from scratchpad.config.topping_config import ToppingType
from scratchpad.model_executor.forward_info import ForwardBatch


class ToppingsManager:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.available_toppings = {}
        for topping in server_args.init_toppings.split(","):
            t_type, t_path, t_name = topping.split(":")
            logger.info(f"Topping: {topping}")
            self.register_topping(t_type, t_path, t_name)
        self.print_available_toppings()
        logger.info("Topping manager ready.")

    def print_available_toppings(self):
        logger.info("Available toppings:")
        for topping in self.available_toppings:
            logger.info(f"({self.available_toppings[topping][0]}) {topping}")

    def prepare_topping_batch(self, forward_batch: ForwardBatch):
        pass

    def register_topping(
        self, topping_type: ToppingType, topping_path: str, topping_name: str
    ):
        self.available_toppings[topping_name] = (topping_type, topping_path)

    def allocate_memory(self):
        pass

    def load_topping(self):
        pass

    def _load_lora(self):
        pass

    def _load_delta(self):
        pass

    def unload_topping(self):
        pass

    def match_target_modules(self, module_name):
        for target_module in self.target_modules:
            if module_name.split(".")[-1] == target_module:
                return True
        return False

    def get_target_modules(self):
        modules = []
        for module_name, module in self.base_model.named_modules():
            if self.match_target_modules(module_name):
                modules.append((module_name, module))
        return modules
