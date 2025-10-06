# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM inference simulator that models the performance of Large Language Model inference on different hardware configurations. The simulator uses roofline analysis to estimate inference times and handles request scheduling across multiple engines.

## Key Commands

### Running Simulations
```bash
# Run the main simulator with default parameters
python cli/start_simulator.py --input <trace_file> --n-engines <num_engines> --arrival-rate <rate>

# Example with typical parameters
python cli/start_simulator.py --input trace.json --n-engines 4 --arrival-rate 1.0

# Output files are generated in .local/replay_results/ by default:
# - trace.json: Chrome trace format events for visualization
# - stats.json: Request statistics and performance metrics
```

### Roofline Analysis
```bash
# Generate roofline plots for different hardware
python cli/plot_roofline.py --hardware nvidia_A100 --matrix-dims 1024 2048 4096 --input-sizes 1 4 16 64

# Save plot to file
python cli/plot_roofline.py --hardware nvidia_A100 --output roofline.png
```

### Dependencies
```bash
pip install -r requirements.txt
# Requires: humanize, transformers, plus standard scientific Python stack (numpy, matplotlib, seaborn)
```

## Architecture Overview

### Core Components

1. **LLMGlobalEngine** (`core/global_engine.py`): Central orchestrator that manages multiple LLM engines, handles request scheduling, and tracks global simulation state.

2. **LLMEngine** (`core/engine.py`): Individual inference engine that processes requests through prefill and decode phases, manages memory allocation, and generates trace events.

3. **GenerationRequest** (`core/request.py`): Represents a single inference request with metadata like input/output lengths, arrival time, and current status.

4. **ModelAnalyzer** (`internal/analyzer/model_analyzer.py`): Performs roofline analysis to estimate inference times based on hardware parameters and model configurations.

5. **Routing Policies** (`core/policies/`): Determines how requests are assigned to engines. Currently implements random routing, with extensible base class for other policies.

### Key Data Flow

1. Requests are loaded from trace files using `utils/loader.py`
2. `LLMGlobalEngine` manages a global queue of pending requests
3. Routing policies assign requests to specific engines based on model compatibility
4. Each engine processes requests through prefill (prompt processing) and decode (token generation) phases
5. Memory planner (`core/memory_planner.py`) tracks GPU memory allocation for KV cache and activations
6. Performance analysis uses roofline model to estimate computation times based on arithmetic intensity

### Hardware Configuration

Hardware parameters are defined in `internal/configs/hardware_params.py` and include:
- Memory bandwidth (bytes/s)
- Peak compute performance (FLOPS/s) for FP16 and INT8
- On-chip buffer size
- Total GPU memory size

Supported hardware includes NVIDIA V100, A100, H100, A6000, L40S, and variants.

### Model Configuration

Model architectures are configured in `internal/configs/llama.py` with functions that extract model parameters like:
- Number of attention heads and layers
- Hidden size and intermediate dimensions
- Layer normalization positions
- Linear layer mappings

The system is designed to work with transformer models from HuggingFace, particularly Llama variants.

## Important Implementation Details

### Memory Management
- KV cache allocation is tracked per-request using the memory planner
- Memory is allocated during prefill and released when requests complete
- Block-based allocation prevents memory fragmentation

### Timing and Synchronization
- Each engine maintains its own timer but advances based on global simulation time
- Events are timestamped in microseconds for Chrome trace compatibility
- The simulation progresses by finding the minimum next event time across all engines

### Request Lifecycle
1. **PENDING**: Request loaded but not yet assigned to engine
2. **SCHEDULED**: Assigned to an engine waiting queue
3. **PREFILL**: Prompt processing phase
4. **GENERATE**: Token-by-token generation
5. **EXIT**: Request completed (either success or failure)

### Performance Modeling
The roofline analysis calculates:
- Arithmetic intensity (FLOPs per byte of memory access)
- Memory-bound vs compute-bound performance
- Inference time estimates for prefill and decode phases
- Memory consumption for weights, activations, and KV cache

## Testing and Validation

The simulator outputs:
- Chrome trace format files for performance visualization
- JSON statistics with latency, throughput, and queue metrics
- SLO pass rates for multi-stage request processing

Typical validation involves comparing simulated latencies against real hardware measurements for known models and hardware configurations.