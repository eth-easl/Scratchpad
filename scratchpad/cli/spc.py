import typer

app = typer.Typer()


@app.command()
def chat(
    model: str,
    backend: str = "http://localhost:3000",
):
    from .handlers import ChatHandler

    print(f"Chatting with model: {model}, backend: {backend}")
    chat_handler = ChatHandler(server_addr=backend, model_name=model)
    chat_handler.chat()


@app.command()
def version():
    """Print the version"""
    typer.echo("0.1.0")
