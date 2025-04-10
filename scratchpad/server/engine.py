import gc
import torch
import asyncio
import multiprocessing as mp
from typing import List, Union, Optional, Dict, Iterator
from scratchpad.managers.structs import GenerateReqInput
from .args import ServerArgs


class AsyncLLMEngine:
    def __init__(self, model_name: str, args: ServerArgs):
        args.model_path = model_name
        args.translate_auto()
        self.args = args
        from scratchpad.managers import TokenizerManager

        self.tokenizer_manager = TokenizerManager(args)
        self.processes: List[mp.Process] = []
        self.loop = asyncio.get_event_loop()
        self._launch()

    def _launch(self):
        mp.set_start_method("spawn", force=True)
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = self.args.tp_size // self.args.nnodes
        tp_rank_range = range(
            tp_size_per_node * self.args.node_rank,
            tp_size_per_node * (self.args.node_rank + 1),
        )
        from scratchpad.scheduler.scheduler import run_scheduler_process

        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(self.args, gpu_id, tp_rank, writer),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        if self.args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            while True:
                pass
        from scratchpad.managers import run_detokenizer_process

        detoken_proc = mp.Process(
            target=run_detokenizer_process,
            args=(self.args,),
        )
        detoken_proc.start()

        for i in range(len(scheduler_pipe_readers)):
            scheduler_pipe_readers[i].recv()
        self.processes = scheduler_procs + [detoken_proc]

    async def generate_request(self, obj: GenerateReqInput):
        try:
            ret = await self.tokenizer_manager.generate_request(obj, None).__anext__()
            return ret
        except ValueError as e:
            return str(e)

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also scratchpad/nn/mm_utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        topping_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            topping_path=topping_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    def generate_chat(
        self,
        messages: List[str],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        prompts = self.tokenizer_manager.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(
            prompt=prompts,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
        )

    def shutdown(self):
        for proc in self.processes:
            proc.terminate()
        for proc in self.processes:
            proc.join()
        gc.collect()
        torch.cuda.empty_cache()
        del self
