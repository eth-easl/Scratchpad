from dataclasses import dataclass
import inspect

import typer
import yaml

@dataclass
class ServerArgs:
    host: str="0.0.0.0"
    port: int=3000
    debug: bool=False
