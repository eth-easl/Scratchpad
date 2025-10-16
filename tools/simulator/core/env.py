import json
from typing import List
from dataclasses import dataclass


@dataclass
class GPUPrecision:
    """Precision configuration for a GPU."""

    weight_bits: int
    activation_bits: int
    kv_bits: int


@dataclass
class GPUConfig:
    """Configuration for a GPU type."""

    name: str
    amount: int
    model: str
    precision: GPUPrecision


@dataclass
class InfrastructureConfig:
    """Infrastructure configuration."""

    disk_bandwidth_mbps: float
    pcie_bandwidth_gbps: float
    network_bandwidth_gbps: float


@dataclass
class ModelLoadingConfig:
    """Model loading configuration."""

    model_cache_path: str
    preload_models: bool


@dataclass
class NodeConfig:
    """Configuration for a compute node with multiple GPUs."""

    node_id: int
    name: str
    gpu_types: List[str]  # Types of GPUs in this node
    gpu_count_per_type: List[int]  # Number of GPUs of each type
    memory_per_gpu_gb: List[float]  # Memory per GPU type
    network_bandwidth_gbps: float


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""

    description: str
    nodes: List[NodeConfig]  # New node-based configuration
    gpus: List[GPUConfig]  # Legacy GPU configuration for backward compatibility
    infrastructure: InfrastructureConfig
    model_loading: ModelLoadingConfig


def load_environment_config(env_file: str) -> EnvironmentConfig:
    """
    Load initial environment configuration from a JSON file.

    Args:
        env_file: Path to the JSON file containing environment configuration

    Returns:
        EnvironmentConfig: Parsed environment configuration

    Raises:
        FileNotFoundError: If the environment file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If required fields are missing or invalid
    """
    try:
        with open(env_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in environment file: {e}")

    # Validate required fields
    if "infrastructure" not in data:
        raise ValueError("Missing 'infrastructure' field in environment configuration")
    if "model_loading_config" not in data:
        raise ValueError(
            "Missing 'model_loading_config' field in environment configuration"
        )

    # Parse nodes (new format) or GPUs (legacy format)
    nodes = []
    gpus = []

    if "nodes" in data:
        # New node-based configuration
        for i, node_data in enumerate(data["nodes"]):
            if "node_id" not in node_data:
                raise ValueError(f"Node config {i}: missing 'node_id' field")
            if "name" not in node_data:
                raise ValueError(f"Node config {i}: missing 'name' field")
            if "gpu_types" not in node_data:
                raise ValueError(f"Node config {i}: missing 'gpu_types' field")
            if "gpu_count_per_type" not in node_data:
                raise ValueError(f"Node config {i}: missing 'gpu_count_per_type' field")
            if "memory_per_gpu_gb" not in node_data:
                raise ValueError(f"Node config {i}: missing 'memory_per_gpu_gb' field")
            if "network_bandwidth_gbps" not in node_data:
                raise ValueError(
                    f"Node config {i}: missing 'network_bandwidth_gbps' field"
                )

            if len(node_data["gpu_types"]) != len(
                node_data["gpu_count_per_type"]
            ) or len(node_data["gpu_types"]) != len(node_data["memory_per_gpu_gb"]):
                raise ValueError(
                    f"Node config {i}: gpu_types, gpu_count_per_type, and memory_per_gpu_gb must have the same length"
                )

            node = NodeConfig(
                node_id=node_data["node_id"],
                name=node_data["name"],
                gpu_types=node_data["gpu_types"],
                gpu_count_per_type=node_data["gpu_count_per_type"],
                memory_per_gpu_gb=node_data["memory_per_gpu_gb"],
                network_bandwidth_gbps=node_data["network_bandwidth_gbps"],
            )
            nodes.append(node)

    # Legacy GPU configuration for backward compatibility
    if "gpus" in data:
        for i, gpu_data in enumerate(data["gpus"]):
            if "name" not in gpu_data:
                raise ValueError(f"GPU config {i}: missing 'name' field")
            if "amount" not in gpu_data:
                raise ValueError(f"GPU config {i}: missing 'amount' field")
            if "model" not in gpu_data:
                raise ValueError(f"GPU config {i}: missing 'model' field")
            if "precision" not in gpu_data:
                raise ValueError(f"GPU config {i}: missing 'precision' field")

            precision_data = gpu_data["precision"]
            if "weight_bits" not in precision_data:
                raise ValueError(f"GPU config {i}: missing 'weight_bits' in precision")
            if "activation_bits" not in precision_data:
                raise ValueError(
                    f"GPU config {i}: missing 'activation_bits' in precision"
                )
            if "kv_bits" not in precision_data:
                raise ValueError(f"GPU config {i}: missing 'kv_bits' in precision")

            precision = GPUPrecision(
                weight_bits=precision_data["weight_bits"],
                activation_bits=precision_data["activation_bits"],
                kv_bits=precision_data["kv_bits"],
            )

            gpu = GPUConfig(
                name=gpu_data["name"],
                amount=gpu_data["amount"],
                model=gpu_data["model"],
                precision=precision,
            )
            gpus.append(gpu)

    # Parse infrastructure configuration
    infra_data = data["infrastructure"]
    if "disk_bandwidth_mbps" not in infra_data:
        raise ValueError("Missing 'disk_bandwidth_mbps' in infrastructure")
    if "pcie_bandwidth_gbps" not in infra_data:
        raise ValueError("Missing 'pcie_bandwidth_gbps' in infrastructure")
    if "network_bandwidth_gbps" not in infra_data:
        raise ValueError("Missing 'network_bandwidth_gbps' in infrastructure")

    infrastructure = InfrastructureConfig(
        disk_bandwidth_mbps=infra_data["disk_bandwidth_mbps"],
        pcie_bandwidth_gbps=infra_data["pcie_bandwidth_gbps"],
        network_bandwidth_gbps=infra_data["network_bandwidth_gbps"],
    )

    # Parse model loading configuration
    loading_data = data["model_loading_config"]
    if "model_cache_path" not in loading_data:
        raise ValueError("Missing 'model_cache_path' in model_loading_config")
    if "preload_models" not in loading_data:
        raise ValueError("Missing 'preload_models' in model_loading_config")

    model_loading = ModelLoadingConfig(
        model_cache_path=loading_data["model_cache_path"],
        preload_models=loading_data["preload_models"],
    )

    description = data.get("description", "Environment configuration")

    config = EnvironmentConfig(
        description=description,
        nodes=nodes,
        gpus=gpus,
        infrastructure=infrastructure,
        model_loading=model_loading,
    )

    print(f"Loaded environment configuration: {description}")
    if nodes:
        total_gpus_in_nodes = sum(sum(node.gpu_count_per_type) for node in nodes)
        print(f"  Nodes: {len(nodes)} nodes, {total_gpus_in_nodes} total GPUs")
    if gpus:
        print(f"  Legacy GPUs: {len(gpus)} types, {sum(g.amount for g in gpus)} total")
    print(f"  Disk bandwidth: {infrastructure.disk_bandwidth_mbps} MB/s")
    print(f"  PCIe bandwidth: {infrastructure.pcie_bandwidth_gbps} GB/s")
    print(f"  Network bandwidth: {infrastructure.network_bandwidth_gbps} GB/s")

    return config


def calculate_model_loading_time(
    model_size_gb: float,
    disk_bandwidth_mbps: float,
    pcie_bandwidth_gbps: float,
    cache_available: bool = False,
) -> float:
    """
    Calculate model loading time based on storage and transfer bandwidths.

    Args:
        model_size_gb: Size of the model in GB
        disk_bandwidth_mbps: Disk read bandwidth in MB/s
        pcie_bandwidth_gbps: PCIe transfer bandwidth in GB/s
        cache_available: Whether model is already cached in GPU memory

    Returns:
        float: Loading time in seconds
    """
    if cache_available:
        return 0.0  # Model already loaded

    # Convert bandwidths to GB/s for consistent units
    disk_bandwidth_gbps = disk_bandwidth_mbps / 1000

    # Loading time is bottlenecked by the slower of disk -> PCIe -> GPU
    # In reality, it's: disk -> system memory -> PCIe -> GPU memory
    # Simplified model: max(disk_transfer_time, pcie_transfer_time)

    disk_time = model_size_gb / disk_bandwidth_gbps
    pcie_time = model_size_gb / pcie_bandwidth_gbps

    # The actual time is the sum of sequential transfers
    total_time = disk_time + pcie_time

    return total_time


@dataclass
class EnvironmentChange:
    """Represents a change in the environment at a specific simulation time."""

    timestamp: float  # Time in seconds when the change occurs
    gpu_name: str  # Action type, e.g., "add_gpu", "remove_gpu"
    amount: dict  # Additional details about the change


def load_environment_changes(change_file: str) -> List[EnvironmentChange]:
    """
    Load dynamic environment changes from a JSONL file.

    Args:
        change_file: Path to the JSONL file containing environment changes

    Returns:
        List[EnvironmentChange]: A list of environment changes
    """
    changes = []
    with open(change_file, "r") as f:
        for line in f:
            data = json.loads(line)
            change = EnvironmentChange(
                timestamp=data["timestamp"],
                gpu_name=data["gpu_name"],
                amount=data["amount"],
            )
            changes.append(change)
    return changes
