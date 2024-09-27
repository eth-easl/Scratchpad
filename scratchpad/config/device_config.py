import torch
from typing import Optional
from scratchpad.utils import current_platform


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if current_platform.is_cuda_alike():
                self.device_type = "cuda"
            elif current_platform.is_cpu():
                self.device_type = "cpu"
            else:
                raise RuntimeError("Failed to infer device type")
        else:
            # Device type is assigned explicitly
            self.device_type = device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron", "openvino"]:
            self.device = torch.device("cpu")
        elif self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)
