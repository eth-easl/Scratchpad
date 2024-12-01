from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scratchpad.server.args import ServerArgs


class ToppingMemPool:
    def __init__(self, args: "ServerArgs"):
        self.args = args
        pass
