from typing import List


def parse_topping_config(topping_conf: str) -> List:
    """
    parse topping_conf
    topping_conf: common-separated string
    """
    toppings = []
    if not topping_conf:
        return None
    topping_confs = topping_conf.split(",")

    for topping_conf in topping_confs:
        tc = topping_conf.split(":")
        if len(tc) == 2:
            topping_type, topping_name, served_name = tc[0], tc[1], tc[1]
        elif len(tc) == 3:
            topping_type, topping_name, served_name = tc[0], tc[1], tc[2]
        else:
            raise ValueError("Invalid topping config: %s" % topping_conf)
        toppings.append((topping_type, topping_name, served_name))
    return toppings
