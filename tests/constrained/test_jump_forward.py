import outlines
import unittest
from scratchpad.constrained.jump_forward import JumpForwardMap, IP_REGEX


def test_main(regex_string):
    jump_forward_map = JumpForwardMap(regex_string)
    for state, e in jump_forward_map.state_to_jump_forward.items():
        if e.symbol is not None:
            jump_forward_str, next_state = jump_forward_map.jump_forward_symbol(state)
            print(f"{state} -> {next_state}", jump_forward_str)
        bytes_ = jump_forward_map.jump_forward_byte(state)
        print(f"{state} -> {bytes_[-1][1]}", [hex(b) for b, _ in bytes_])


if __name__ == "__main__":
    import outlines

    outlines.caching.clear_cache()
    test_main(r"The google's DNS sever address is " + IP_REGEX)
    test_main(r"霍格沃茨特快列车|霍比特人比尔博")
    # 霍格: \xe9\x9c\x8d \xe6\xa0\xbc ...
    # 霍比: \xe9\x9c\x8d \xe6\xaf\x94 ...

    test_main(r"[-+]?[0-9]+[ ]*")
