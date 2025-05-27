import typer
from scratchpad.server import dataclass_to_cli, ServerArgs
from .handlers import ChatHandler, benchmark_quality
import torch.distributed as dist

app = typer.Typer()


@app.command()
@dataclass_to_cli
def serve(
    model: str,
    args: ServerArgs,
):
    """Spin up the server"""
    from scratchpad.server import launch_server
    import multiprocessing as mp
    import signal
    import sys

    def signal_handler(sig, frame):
        typer.echo("\nReceived interrupt. Shutting down server gracefully...")
        # Kill all child processes
        dist.destroy_process_group()
        typer.echo("Server shutdown complete.")
        sys.exit(0)

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    mp.set_start_method("spawn", force=True)
    typer.echo(f"Serving model: {model}, args: {args}")

    try:
        launch_server(model, args)
    except KeyboardInterrupt:
        typer.echo("\nShutting down server gracefully...")
        typer.echo("Server shutdown complete.")
    except Exception as e:
        typer.echo(f"Error: {e}")
        sys.exit(1)


@app.command()
def version():
    """Print the version"""
    typer.echo("0.1.0")


@app.command()
def chat(
    model: str,
    backend: str = "http://localhost:3000",
):
    chat_handler = ChatHandler(server_addr=backend, model_name=model)
    chat_handler.chat()


@app.command()
def benchmark(
    model: str,
    tasks: str = "mmlu",
    url: str = "http://localhost:8080/v1",
    num_fewshot: int = 0,
    instruct_model: bool = False,
):
    url = url + "/completions"
    benchmark_quality(model, url, tasks, num_fewshot, instruct_model)


if __name__ == "__main__":
    app()
