# Repository Guidelines

## Project Structure & Module Organization
- `core/` holds the scheduling engines and memory planners; start with `core/global_engine.py` or `core/node_global_engine.py` when altering execution flow.
- `cli/` provides runnable entry points such as `run_simulator.py` (simulation driver) and `plot_roofline.py` (visualization helper).
- `api/` and `utils/` supply service adapters and shared helpers—trace loading, hardware math, serializers—so prefer adding reusable logic there instead of duplicating it.
- `internal/` packages the analyzer toolkit plus canonical hardware configs; treat these files as reference data.
- `examples/` stores sample traces and environment JSON/JSONL fixtures for smoke tests; generated outputs belong in `.local/` and stay untracked.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` to isolate dependencies.
- `python -m pip install -r requirements.txt` installs runtime packages (`humanize`, `transformers`).
- `python cli/run_simulator.py --input examples/trace.jsonl --n-engines 2 --arrival-rate 1.5 --trace-output .local/trace.json --stats-output .local/stats.json` runs the canonical workload and surfaces performance statistics.
- `python cli/plot_roofline.py --input .local/stats.json --out plots/roofline.png` turns stats into an image; create the destination directory first.

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, and PEP 8 naming (snake_case for modules/functions, UpperCase for enums like `REQ_STATUS`).
- Use type hints and concise docstrings similar to `core/request.py:GenerationRequest` to clarify intent.
- Group imports stdlib → third-party → local, and expose public symbols explicitly in `__init__.py` files when it improves discoverability.

## Testing Guidelines
- No automated suite exists yet; replay `cli/run_simulator.py` with `examples` fixtures and inspect `.local/stats.json` for regressions after each change.
- New tests should rely on `pytest` under `tests/` mirroring the module structure (e.g., `tests/core/test_memory_planner.py`) with descriptive names like `test_allocates_kv_cache`.
- Capture before/after throughput or latency figures when altering performance-sensitive code and share them in the review thread.

## Commit & Pull Request Guidelines
- Follow the history style: imperative subjects (`Add README for LLM Simulator`) and optional issue references in parentheses (e.g., `(#60)`).
- Keep commits focused and avoid checking in artifacts from `.local/` or large trace files.
- Pull requests need a short motivation, the commands you ran (build/test), and links or screenshots for visualization changes.

## Security & Configuration Tips
- Review `internal/configs/hardware_params.py` and `examples/env.json` before adding hardware profiles; never commit production-specific credentials.
- Treat environment-change JSONL fixtures as append-only—add new files for new scenarios instead of rewriting shared samples.
