from tools.simulator.config.hardware_params import hardware_params
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns


def calculate_roofline_data(
    hw_params: Dict, matrix_dims: List[Tuple[int, int, int]], input_sizes: List[int]
) -> Dict:
    """
    Calculate data points for the roofline model based on hardware parameters,
    matrix dimensions, and input sizes.

    Args:
        hw_params: Dictionary of hardware parameters
        matrix_dims: List of matrix dimensions to consider as (M, N, K) tuples
                     where M×K multiplied by K×N gives M×N result
        input_sizes: List of input/batch sizes to consider

    Returns:
        Dictionary containing data for the roofline plot
    """
    peak_compute = hw_params["FP16"]  # Peak computational performance in FLOPS
    peak_memory_bandwidth = hw_params["bandwidth"]  # Memory bandwidth in bytes/sec

    # Calculate arithmetic intensities and performance for each combination
    results = {
        "arithmetic_intensities": [],
        "performance_limits": [],
        "matrix_input_pairs": [],
    }

    for matrix_dim in matrix_dims:
        m, n, k = matrix_dim
        for input_size in input_sizes:
            # Compute arithmetic intensity (FLOPS/byte)
            element_size = 2  # FP16 = 2 bytes

            # Special case for vectors (either m=1 or n=1)
            if m == 1 or n == 1:
                # For matrix-vector multiplication or vector-matrix multiplication
                # Memory accesses:
                # 1. Load the weight matrix (m*k elements)
                # 2. Load the input vectors (k*batch_size elements)
                # 3. Store the output vectors (m*batch_size elements)
                flops = 2 * m * n * k * input_size  # multiply-add operations

                if m == 1:  # Vector-matrix case (1×K)×(K×N) = (1×N)
                    weights_bytes = m * n * k * element_size  # The weight matrix (1×K)
                    input_bytes = (
                        k * element_size * input_size
                    )  # Input vectors (K×batch_size)
                    output_bytes = (
                        n * element_size * input_size
                    )  # Output vectors (N×batch_size)
                elif n == 1:  # Matrix-vector case (M×K)×(K×1) = (M×1)
                    weights_bytes = m * n * k * element_size  # The weight matrix (M×K)
                    input_bytes = (
                        k * element_size * input_size
                    )  # Input vectors (K×batch_size)
                    output_bytes = (
                        m * element_size * input_size
                    )  # Output vectors (M×batch_size)

                bytes_accessed = weights_bytes + input_bytes + output_bytes
            else:
                # Standard matrix-matrix multiplication
                # Memory accesses: (M*K + K*N + M*N) * element_size * input_size
                flops = 2 * m * n * k * input_size
                bytes_accessed = (m * k + k * n + m * n) * element_size * input_size

            arithmetic_intensity = flops / bytes_accessed
            print(
                f"input_size: {input_size}, m: {m}, n: {n}, k: {k}, flops: {flops}, bytes_accessed: {bytes_accessed}, arithmetic_intensity: {arithmetic_intensity}"
            )
            # Calculate attainable performance (min of peak compute and memory-bound limit)
            memory_bound_limit = peak_memory_bandwidth * arithmetic_intensity
            perf_limit = min(peak_compute, memory_bound_limit)
            time_needed_for_computation = flops / peak_compute
            time_needed_for_data_transfer = bytes_accessed / peak_memory_bandwidth

            print(
                f"Time needed for computation: {time_needed_for_computation}, Time needed for data transfer: {time_needed_for_data_transfer}"
            )

            results["arithmetic_intensities"].append(arithmetic_intensity)
            results["performance_limits"].append(perf_limit)
            results["matrix_input_pairs"].append((matrix_dim, input_size))

    return results


def plot_roofline(
    hw_name: str, data: Dict, log_scale: bool = True, save_path: Optional[str] = None
):
    """
    Plot the roofline model based on the calculated data.

    Args:
        hw_name: Name of the hardware platform
        data: Dictionary containing roofline data
        log_scale: Whether to use logarithmic scales
        save_path: Path to save the figure, if None the figure is displayed
    """
    hw_params = hardware_params[hw_name]
    peak_compute = hw_params["FP16"]
    peak_memory_bandwidth = hw_params["bandwidth"]

    # Calculate ridge point (arithmetic intensity where memory and compute bounds meet)
    ridge_point = peak_compute / peak_memory_bandwidth

    # Create plot - fix the figure and axes creation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Plot the roofline
    if log_scale:
        # X-range for the memory-bound region (up to ridge point)
        x_mem = np.logspace(-2, np.log10(ridge_point), 100)
        # Constant value for compute-bound region (after ridge point)
        x_compute = np.logspace(np.log10(ridge_point), 3, 100)
    else:
        x_mem = np.linspace(0.01, ridge_point, 100)
        x_compute = np.linspace(ridge_point, 1000, 100)

    y_mem = peak_memory_bandwidth * x_mem
    y_compute = np.ones_like(x_compute) * peak_compute

    ax.plot(x_mem, y_mem, "b-", linewidth=4, label="Memory-bound")
    ax.plot(x_compute, y_compute, "r-", linewidth=4, label="Compute-bound")

    # Group data points by matrix dimensions for better visualization
    matrix_dims = {}
    for ai, perf, ((m, n, k), input_size) in zip(
        data["arithmetic_intensities"],
        data["performance_limits"],
        data["matrix_input_pairs"],
    ):
        # Fix the matrix key to use consistent dimension order (M×K×N)
        matrix_key = f"{m}×{k}×{n}"
        if matrix_key not in matrix_dims:
            matrix_dims[matrix_key] = {"ai": [], "perf": [], "input_sizes": []}

        matrix_dims[matrix_key]["ai"].append(ai)
        matrix_dims[matrix_key]["perf"].append(perf)
        matrix_dims[matrix_key]["input_sizes"].append(input_size)

    # Use different markers and colors for different matrix dimensions
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "+", "x"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(matrix_dims))))

    # Plot data points with different markers for different matrix dimensions
    # Fix the legend to avoid duplicates
    legend_added = set()
    for i, (matrix_key, values) in enumerate(matrix_dims.items()):
        for j, (ai, perf, input_size) in enumerate(
            zip(values["ai"], values["perf"], values["input_sizes"])
        ):
            label = None
            matrix_label = f"{matrix_key}, Input: {input_size}"
            if matrix_label not in legend_added:
                label = matrix_label
                legend_added.add(matrix_label)

            ax.scatter(
                ai,
                perf,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                s=100 + j * 25,  # Larger size for larger input sizes
                alpha=0.8,
                label=label,
            )

    # Set scales
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Ridge point annotation
    ax.scatter(
        [ridge_point],
        [peak_compute],
        marker="*",
        s=200,
        color="green",
        label=f"Ridge Point: {ridge_point:.2f}",
    )

    # Labels and title
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=14)
    ax.set_ylabel("Attainable Performance (FLOPS/s)", fontsize=14)
    ax.set_title(
        f"Roofline Model for {hw_name}\nPeak Compute: {peak_compute/1e12:.1f} TFLOPS, "
        f"Memory BW: {peak_memory_bandwidth/1e9:.1f} GB/s",
        fontsize=16,
    )

    # Grid and legend
    ax.grid(True, which="both", ls="--", alpha=0.7)

    # Improve legend with smaller font and better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=10,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=2,
    )

    plt.tight_layout()
    sns.despine()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def parse_matrix_dimensions(dimensions_str: List[str]) -> List[Tuple[int, int, int]]:
    """
    Parse matrix dimensions from string format.

    Args:
        dimensions_str: List of strings in format 'MxKxN' or 'NxN' for square matrices

    Returns:
        List of tuples representing (M, N, K) dimensions
    """
    result = []
    for dim_str in dimensions_str:
        parts = dim_str.lower().split("x")
        if len(parts) == 1:  # Just a single number N for square NxN matrix
            n = int(parts[0])
            result.append((n, n, n))  # M=N=K for square matrices
        elif len(parts) == 2:  # MxN format (assuming K = N)
            m, n = map(int, parts)
            result.append((m, n, n))
        elif len(parts) == 3:  # MxKxN format
            m, k, n = map(int, parts)
            result.append((m, n, k))
        else:
            raise ValueError(
                f"Invalid matrix dimension format: {dim_str}. Use 'MxKxN', 'MxN' or just 'N'"
            )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf")
    parser.add_argument(
        "--hardware",
        type=str,
        default="nvidia_A100",
        choices=list(hardware_params.keys()),
        help="Hardware platform to use for roofline model",
    )
    parser.add_argument(
        "--matrix-dims",
        type=str,
        nargs="+",
        default=["1024", "2048", "4096", "8192"],
        help="Matrix dimensions to consider. Can be specified as 'N' for N×N×N, 'M×N' for M×N×N, or 'M×K×N' for full specification",
    )
    parser.add_argument(
        "--input-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Input/batch sizes to consider",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Use logarithmic scale for the plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot (if not provided, plot will be displayed)",
    )

    args = parser.parse_args()

    # Validate hardware choice
    if args.hardware not in hardware_params:
        print(f"Error: Hardware '{args.hardware}' not found in hardware_params.")
        print(f"Available options: {list(hardware_params.keys())}")
        exit(1)

    # Parse matrix dimensions
    matrix_dims = parse_matrix_dimensions(args.matrix_dims)

    # Calculate roofline data
    roofline_data = calculate_roofline_data(
        hardware_params[args.hardware], matrix_dims, args.input_sizes
    )

    # Plot the roofline
    plot_roofline(args.hardware, roofline_data, args.log_scale, args.output)
