import bisect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch
from scratchpad.distributed.parallel_state import graph_capture
from .custom_op import CustomOp

from scratchpad.nn.layers.fused_moe.patch import fused_moe_forward_native
from scratchpad.nn.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from scratchpad.scheduler.schedule_batch import ScheduleBatch
from .forward_info import ForwardMode, InputMetadata
from scratchpad.utils import monkey_patch_vllm_all_gather

if TYPE_CHECKING:
    from .model_runner import ModelRunner
    from scratchpad.distributed.parallel_state import GroupCoordinator


def _to_torch(model: torch.nn.Module, reverse: bool = False):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse)


@contextmanager
def patch_model(
    model: torch.nn.Module, enable_compile: bool, tp_group: "GroupCoordinator"
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model)
            monkey_patch_vllm_all_gather()
            backup_ca_comm = tp_group.ca_comm
            tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward), mode="max-autotune-no-cudagraphs"
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True)
            monkey_patch_vllm_all_gather(reverse=True)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: "ModelRunner"):
        # Parse args
        self.model_runner = model_runner
        self.graphs = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.flashinfer_handlers = {}
        self.graph_memory_pool = None
        self.use_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding

        # Batch sizes to capture
        if self.model_runner.server_args.disable_cuda_graph_padding:
            self.capture_bs = list(range(1, 32)) + [64, 128]
        else:
            self.capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]

        self.capture_bs = [
            bs for bs in self.capture_bs if bs <= model_runner.req_to_token_pool.size
        ]
        self.compile_bs = (
            [
                bs
                for bs in self.capture_bs
                if bs <= self.model_runner.server_args.max_torch_compile_bs
            ]
            if self.use_torch_compile
            else []
        )

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs)
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        if self.use_torch_compile:
            set_torch_compile_config()

        # Common inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.position_ids_offsets = torch.ones((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.zeros((self.max_bs,), dtype=torch.int32)

        # Capture
        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
            )

    def can_run(self, batch_size: int):
        if self.disable_padding:
            return batch_size in self.graphs
        else:
            return batch_size <= self.max_bs

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            for bs in self.capture_bs:
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output_buffers

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Common inputs
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        position_ids_offsets = self.position_ids_offsets[:bs]
        out_cache_loc = self.out_cache_loc[:bs]

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs, req_pool_indices, seq_lens
        )

        # Run and capture
        def run_once():
            input_metadata = InputMetadata(
                forward_mode=ForwardMode.DECODE,
                batch_size=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                return_logprob=False,
                top_logprobs_nums=[0] * bs,
                positions=(seq_lens - 1 + position_ids_offsets).to(torch.int64),
            )
            return forward(input_ids, input_metadata.positions, input_metadata)

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        self.graph_memory_pool = graph.pool()
        return graph, out

    def replay(self, batch: ScheduleBatch):
        assert batch.out_cache_loc is not None
        raw_bs = len(batch.reqs)

        # Pad
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.position_ids_offsets.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs] = batch.input_ids
        self.req_pool_indices[:raw_bs] = batch.req_pool_indices
        self.seq_lens[:raw_bs] = batch.seq_lens
        self.position_ids_offsets[:raw_bs] = batch.position_ids_offsets
        self.out_cache_loc[:raw_bs] = batch.out_cache_loc

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs, self.req_pool_indices, self.seq_lens
        )

        # Replay
        self.graphs[bs].replay()
        logits_output = self.output_buffers[bs]

        # Unpad
        if bs != raw_bs:
            logits_output = LogitsProcessorOutput(
                next_token_logits=logits_output.next_token_logits[:raw_bs],
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )

        # Extract logprobs
        if batch.return_logprob:
            logits_output.next_token_logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits, dim=-1
            )
            return_top_logprob = any(x > 0 for x in batch.top_logprobs_nums)
            if return_top_logprob:
                logits_metadata = LogitsMetadata(
                    forward_mode=ForwardMode.DECODE,
                    top_logprobs_nums=batch.top_logprobs_nums,
                )
                logits_output.output_top_logprobs = LogitsProcessor.get_top_logprobs(
                    logits_output.next_token_logprobs, logits_metadata
                )[1]

        return logits_output
