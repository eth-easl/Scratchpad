hardware_params = {
    "nvidia_A100": {
        "bandwidth": 1555e9,
        "FP16": 312e12,
        "FP16_sparse": 624e12,
        "onchip_buffer": 27648e3,
    },  # use 40G data
    "nvidia_A100_40G": {
        "bandwidth": 1555e9,
        "FP16": 312e12,
        "FP16_sparse": 624e12,
        "onchip_buffer": 27648e3,
    },
    "nvidia_A100_80G": {
        "bandwidth": 2039e9,
        "FP16": 312e12,
        "FP16_sparse": 624e12,
        "onchip_buffer": 27648e3,
    },
    "nvidia_A40": {
        "bandwidth": 696e9,
        "FP16": 149.7e12,
        "FP16_sparse": 299.3e12,
        "onchip_buffer": 21504e3,
    },
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "FP16_sparse": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },  # use SXM data
    "nvidia_H100_SXM": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },
    "nvidia_H100_PCIe": {
        "bandwidth": 2048e9,
        "FP16": 1513e12 / 2,
        "INT8": 3026e12 / 2,
        "onchip_buffer": 29184e3,
    },
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {
        "bandwidth": 864e9,
        "FP16": 181e12,
        "INT8": 362e12,
        "onchip_buffer": 36352e3,
    },
}
