from typing import TYPE_CHECKING


class EngineStates:
    def __init__(self):
        self.served_extra_models = []

    def add_toppings(self, topping_name: str, topping_type: str):
        self.served_extra_models.append(topping_name)

    @property
    def toppings(self):
        return self.served_extra_models
