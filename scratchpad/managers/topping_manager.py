from scratchpad.server.args import ServerArgs


class ToppingManager:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        pass

    def prepare_topping_batch(self):
        pass

    def register_topping(self, topping_path):
        pass

    def allocate_memory(self):
        pass

    def load_topping(self):
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

    def set_topping_module(self, module_name, module, topping_module):
        pass
