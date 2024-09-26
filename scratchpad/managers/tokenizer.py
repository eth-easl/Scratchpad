import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from scratchpad.distributed import *