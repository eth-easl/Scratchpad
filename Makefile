format:
	python -m black .
cli-docs:
	typer scratchpad.cli.sp utils docs --name scratchpad --output docs/cli.md
container:
	bash docker/build_image.sh $(version)