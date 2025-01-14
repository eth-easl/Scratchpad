from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .structs import RegisterToppingsReqInput


class EngineStates:
    def __init__(self):
        self.root_model: str = ""
        self.served_extra_models = []

    def add_toppings(self, req: "RegisterToppingsReqInput"):
        self.served_extra_models.append(req.served_name)

    @property
    def toppings(self):
        return self.served_extra_models

    def get_topping_path(self, topping_name) -> Optional[str]:
        if topping_name == self.root_model:
            return None
        else:
            return topping_name
