from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .structs import RegisterToppingsReqInput


class EngineStates:
    def __init__(self):
        self.served_extra_models = []

    def add_toppings(self, req: "RegisterToppingsReqInput"):
        self.served_extra_models.append(req.served_name)

    @property
    def toppings(self):
        return self.served_extra_models
