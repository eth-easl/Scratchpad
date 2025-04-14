import inspect
import yaml
import typer
import dataclasses
from transformers import AutoConfig


def conf_callback(ctx: typer.Context, param: typer.CallbackParam, value: str) -> str:
    """
    Callback for typer.Option that loads a config file from the first
    argument of a dataclass.
    Based on https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
    """
    if param.name == "config" and value:
        typer.echo(f"Loading config file: {value}")
        try:
            with open(value, "r") as f:
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}
            ctx.default_map.update(conf)
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return value


def dataclass_to_cli(func):
    """
    Converts a function taking a dataclass as its first argument into a
    dataclass that can be called via `typer` as a CLI.
    Additionally, the --config option will load a yaml configuration before the
    other arguments.
    Modified from:
    - https://github.com/tiangolo/typer/issues/197
    A couple related issues:
    - https://github.com/tiangolo/typer/issues/153
    - https://github.com/tiangolo/typer/issues/154
    """

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)

    param = [x for x in list(sig.parameters.values())]
    cls = [x.annotation for x in param if dataclasses.is_dataclass(x.annotation)][0]
    remaining_params = [x for x in param if x.annotation != cls]

    def wrapped(**kwargs):
        conf = {}
        # CLI options override the config file.
        # if the k in kwargs are in cls, then we update the conf
        dataclass_kwargs = {
            k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__
        }
        conf.update(dataclass_kwargs)
        # Convert back to the original dataclass type.
        arg = cls(**conf)
        # for remaining_params, we need to update the arg with the remaining params
        remaining_params = [
            x for x in kwargs.keys() if x not in cls.__dataclass_fields__
        ]
        # Actually call the entry point function.
        # in the func call, first process remaining_params, then arg
        remaining_params = [kwargs[x] for x in remaining_params]
        return func(*remaining_params, arg)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(cls.__init__)
    parameters = list(signature.parameters.values())
    if len(parameters) > 0 and parameters[0].name == "self":
        del parameters[0]

    # Add the --config option to the signature.
    # When called through the CLI, we need to set defaults via the YAML file.
    # Otherwise, every field will get overwritten when the YAML is loaded.

    parameters = remaining_params + [p for p in parameters if p.name != "config"]

    # The new signature is compatible with the **kwargs argument.
    wrapped.__signature__ = signature.replace(parameters=parameters)
    # The docstring is used for the explainer text in the CLI.
    wrapped.__doc__ = func.__doc__ + "\n" + ""
    wrapped.__name__ = func.__name__
    return wrapped


def register_hf_configs():
    from scratchpad.nn.models.swissai.config import SwissAIConfig

    AutoConfig.register("swissai", SwissAIConfig)
