import torch
import threading
import dataclasses
from typing import TYPE_CHECKING, List, Optional, Dict, Callable, Tuple, Any

from scratchpad.utils import logger
from . import penaltylib
from .custom_logit_processor import CustomLogitProcessor

if TYPE_CHECKING:
    from scratchpad.scheduler.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # All requests use greedy sampling
    is_all_greedy: bool

    # Dispatch in CUDA graph
    need_min_p_sampling: bool
    has_custom_logit_processor: bool

    # Bias Tensors
    vocab_size: int
    logit_bias: torch.Tensor = None
    vocab_mask: Optional[torch.Tensor] = None
    grammars: Optional[List] = None
    sampling_info_done: Optional[threading.Event] = None
    apply_mask: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    linear_penalties: Optional[torch.Tensor] = None
    scaling_penalties: Optional[torch.Tensor] = None

    # Device
    device: str = "cuda"

    # Custom Parameters
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None

    # Custom Logit Processor
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    @classmethod
    def from_schedule_batch(cls, batch: "ScheduleBatch", vocab_size: int):
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            batch.enable_custom_logit_processor  # check the flag first.
            and any(r.custom_logit_processor for r in reqs)  # then check the requests.
        )

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)

            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we can choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
            },
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_penalties(self):
        if not self.penalizer_orchestrator:
            return

        self.scaling_penalties = None
        self.linear_penalties = None

        for penalizer in self.penalizer_orchestrator.penalizers.values():
            if not penalizer.is_prepared():
                continue

            if isinstance(penalizer, penaltylib.BatchedRepetitionPenalizer):
                self.scaling_penalties = penalizer.cumulated_repetition_penalties
            else:
                if self.linear_penalties is None:
                    bs = self.penalizer_orchestrator.batch.batch_size()
                    self.linear_penalties = torch.zeros(
                        (bs, self.vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                self.linear_penalties = penalizer.apply(self.linear_penalties)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask = None
            return

        # find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask = first_grammar.apply_vocab_mask  # force to use static method

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished:
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        self.penalizer_orchestrator.filter(keep_indices_device)

        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices_device])

    def _filter_batch_custom_logit_processor(
        self, unfinished_indices: List[int], new_indices: torch.Tensor
    ):
        """Filter the custom logit processor and custom params"""
        if not self.custom_logit_processor:
            return
        self.custom_logit_processor = {
            k: (p, mask[new_indices])
            for k, (p, mask) in self.custom_logit_processor.items()
            if any(
                mask[new_indices]
            )  # ignore the custom logit processor whose mask is all False
        }
        self.custom_params = [self.custom_params[i] for i in unfinished_indices]

        if len(self) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    @staticmethod
    def merge_bias_tensor(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        bs1: int,
        bs2: int,
        device: str,
        default: int = 0,
    ):
        # bias tensor can be None
        if lhs is not None or rhs is not None:
            shape, dtype = None, None
            if lhs is not None:
                shape, dtype = lhs.shape[1:], lhs.dtype
            else:
                shape, dtype = rhs.shape[1:], rhs.dtype
            with torch.dtype(dtype):
                if lhs is None:
                    lhs = torch.empty((bs1, *shape), device=device).fill_(default)
                if rhs is None:
                    rhs = torch.empty((bs2, *shape), device=device).fill_(default)
            return torch.cat([lhs, rhs])

        return None

    @staticmethod
    def merge_custom_logit_processor(
        lhs: Optional[Dict[str, torch.Tensor]],
        rhs: Optional[Dict[str, torch.Tensor]],
        bs1: int,
        bs2: int,
        device: str,
    ):
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            # Get the logit processor object
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            # Get and merge the mask tensors from the two dicts
            left_mask = (
                lhs[k][1]
                if k in lhs
                else torch.zeros(bs1, dtype=torch.bool, device=device)
            )
            right_mask = (
                rhs[k][1]
                if k in rhs
                else torch.zeros(bs2, dtype=torch.bool, device=device)
            )
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

        return merged_dict

    def merge_batch(self, other: "SamplingBatchInfo"):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.concat([self_val, other_val]))

        self.is_all_greedy = self.is_all_greedy and other.is_all_greedy
        self.logit_bias = SamplingBatchInfo.merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device
        )
        self.need_min_p_sampling = self.need_min_p_sampling or other.need_min_p_sampling

        # Merge the custom logit processors and custom params lists
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            # Merge the custom logit processors
            self.custom_logit_processor = (
                SamplingBatchInfo.merge_custom_logit_processor(
                    self.custom_logit_processor,
                    other.custom_logit_processor,
                    len(self),
                    len(other),
                    self.device,
                )
            )
            # Merge the custom params lists
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            # Set the flag to True if any of the two has custom logit processor
            self.has_custom_logit_processor = True

    def apply_logits_bias(self, logits: torch.Tensor):
        # Apply logit_bias
        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

        # min-token, presence, frequency
        if self.linear_penalties is not None:
            logits.add_(self.linear_penalties)

        # repetition
        if self.scaling_penalties is not None:
            # TODO(xiaozhe): sgl has a cuda kernel for this operation.
            logits[:] = torch.where(
                logits > 0,
                logits / self.scaling_penalties,
                logits * self.scaling_penalties,
            )

        # Apply regex vocab_mask
        if self.vocab_mask is not None:
            self.apply_mask(logits=logits, vocab_mask=self.vocab_mask)
