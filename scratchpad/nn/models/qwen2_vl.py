from functools import lru_cache, partial
from typing import Iterable, List, Optional, Tuple, Type, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from scratchpad.distributed import parallel_state
from scratchpad.distributed import utils as dist_utils
from scratchpad.utils import logger
