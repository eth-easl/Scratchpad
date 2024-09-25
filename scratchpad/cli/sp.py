import typer

app = typer.Typer()

@app.command()
def serve(model: str):
    typer.echo(f"Serving model: {model}")

if __name__ == "__main__":
    app()