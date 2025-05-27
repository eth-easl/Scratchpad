import yaml
import json
import httpx
import typer
import asyncio
import inspect
import dataclasses
from http import HTTPStatus
from typing import TYPE_CHECKING
from transformers import AutoConfig
from scratchpad.utils import logger


if TYPE_CHECKING:
    from .args import ServerArgs


async def run_post_startup_check(server_args: "ServerArgs", tokenizer_manager):
    """Sends a request to /v1/models after server startup to verify server health and API capabilities."""

    initial_wait_seconds = 5.0
    logger.info(
        f"Post-startup check: Initializing. Waiting {initial_wait_seconds}s for Uvicorn to be ready..."
    )
    await asyncio.sleep(initial_wait_seconds)
    logger.info("Post-startup check: Initial wait complete. Starting actual checks.")

    # Initial check for server_args and tokenizer_manager
    if not server_args or not tokenizer_manager:
        logger.error(
            "Post-startup check: Server args or tokenizer manager not initialized. Skipping."
        )
        return

    health_check_url = f"http://localhost:{server_args.port}/health"
    openai_models_check_url = f"http://localhost:{server_args.port}/v1/models"  # Changed to v1/models endpoint
    max_wait_seconds = 60
    poll_interval_seconds = 2
    elapsed_wait_seconds = 0

    logger.info(
        f"Post-startup: Waiting for server to be healthy at {health_check_url}..."
    )

    try:
        async with httpx.AsyncClient() as client:
            server_healthy = False
            while elapsed_wait_seconds < max_wait_seconds:
                try:
                    health_response = await client.get(health_check_url, timeout=5.0)
                    if health_response.status_code == HTTPStatus.OK:
                        logger.info(
                            "Post-startup: Server is healthy. Proceeding with /v1/models check."
                        )
                        server_healthy = True
                        break
                except httpx.RequestError as e:
                    logger.debug(
                        f"Post-startup: Health check attempt to {health_check_url} failed: {e}. Retrying in {poll_interval_seconds}s..."
                    )

                await asyncio.sleep(poll_interval_seconds)
                elapsed_wait_seconds += poll_interval_seconds

            if not server_healthy:
                logger.error(
                    f"Post-startup: Server did not become healthy at {health_check_url} within {max_wait_seconds} seconds. Skipping /v1/models check."
                )
                return

            logger.info(
                f"Post-startup: Running check to /v1/models endpoint at {openai_models_check_url}..."
            )

            headers = {}
            if server_args.api_key:
                headers["X-API-Key"] = server_args.api_key

            try:
                # Changed to GET request to /v1/models, removed payload
                response = await client.get(
                    openai_models_check_url, headers=headers, timeout=30.0
                )

                if response.status_code == HTTPStatus.OK:
                    logger.info(
                        f"Post-startup: /v1/models check successful. Status: {response.status_code}"
                    )
                    try:
                        response_json = response.json()
                        logger.info(
                            f"Post-startup: /v1/models response (first 200 chars): {str(response_json)[:200]}"
                        )

                        # Attempt to get a model ID for the chat completion check
                        model_id_for_chat = None
                        if (
                            isinstance(response_json, dict)
                            and "data" in response_json
                            and isinstance(response_json["data"], list)
                            and len(response_json["data"]) > 0
                        ):
                            first_model_entry = response_json["data"][0]
                            if (
                                isinstance(first_model_entry, dict)
                                and "id" in first_model_entry
                            ):
                                model_id_for_chat = first_model_entry["id"]

                        if model_id_for_chat:
                            logger.info(
                                f"Post-startup: Attempting chat completion check with model '{model_id_for_chat}'."
                            )
                            chat_completion_url = f"http://localhost:{server_args.port}/v1/chat/completions"
                            chat_payload = {
                                "model": model_id_for_chat,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Hello! This is a post-startup test query.",
                                    }
                                ],
                            }

                            try:
                                # Ensure client, headers, and server_args are accessible here
                                chat_response = await client.post(
                                    chat_completion_url,
                                    json=chat_payload,
                                    headers=headers,
                                    timeout=60.0,
                                )
                                if chat_response.status_code == HTTPStatus.OK:
                                    logger.info(
                                        f"Post-startup: /v1/chat/completions check successful. Status: {chat_response.status_code}"
                                    )
                                    try:
                                        chat_response_json = chat_response.json()
                                        logger.info(
                                            f"Post-startup: /v1/chat/completions response (first 200 chars): {str(chat_response_json)[:200]}"
                                        )
                                    except json.JSONDecodeError:
                                        logger.info(
                                            f"Post-startup: /v1/chat/completions response (non-JSON, first 200 chars): {chat_response.text[:200]}"
                                        )
                                else:
                                    logger.error(
                                        f"Post-startup: /v1/chat/completions check failed. Status: {chat_response.status_code}, Response: {chat_response.text[:500]}"
                                    )
                            except httpx.RequestError as chat_req_err:
                                logger.error(
                                    f"Post-startup: HTTPX RequestError during /v1/chat/completions check to {chat_completion_url}: {chat_req_err}"
                                )
                            except Exception as chat_gen_err:
                                logger.error(
                                    f"Post-startup: Unexpected exception during /v1/chat/completions check: {chat_gen_err}",
                                    exc_info=True,
                                )
                        else:
                            logger.warning(
                                "Post-startup: Could not determine a model ID from /v1/models response. Skipping /v1/chat/completions check."
                            )

                    except json.JSONDecodeError:
                        logger.info(
                            f"Post-startup: /v1/models response (non-JSON, first 200 chars): {response.text[:200]}"
                        )
                        logger.warning(
                            "Post-startup: Could not parse /v1/models response as JSON. Skipping /v1/chat/completions check."
                        )
                else:
                    logger.error(
                        f"Post-startup: /v1/models check failed. Status: {response.status_code}, Response: {response.text[:500]}"
                    )
            except httpx.RequestError as e:
                logger.error(
                    f"Post-startup: HTTPX RequestError during /v1/models check to {openai_models_check_url}: {e}"
                )

    except AttributeError as e:
        logger.error(
            f"Post-startup: AttributeError during check (likely server_args or tokenizer_manager not fully ready): {e}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"Post-startup: Unexpected exception during check: {e}", exc_info=True
        )


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
