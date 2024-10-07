help:
	@echo "format - format the code"
	@echo "container - build the docker container"
	@echo "docs - build the documentation"
	@echo "cli-docs - build the cli documentation"
	@echo "help - show this message"
.PHONY: help Makefile

format:
	python -m black .
container:
	bash docker/build_image.sh $(version)
html-docs:
	sphinx-build -M html docs/sources docs/build 
cli-docs:
	typer scratchpad.cli.sp utils docs --title "CLI Reference" --name "scratchpad" --output docs/sources/cli.md 