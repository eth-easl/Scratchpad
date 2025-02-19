import pytest
import unittest
from tests.e2e.hf_utils import clm_generate
from scratchpad.server.engine import AsyncLLMEngine
from scratchpad.server.args import ServerArgs


class TestGenerationResultWithHF(unittest.TestCase):
    def setUp(self):
        self.attention_backends = ["flashinfer"]
        self.sampling_backends = ["flashinfer"]
        # set_random_seed(42)

    def run_generation(
        self,
        hf_model_identifier,
        prompts,
        attention_backend,
        sampling_backend,
        min_new_tokens: int = 50,
        max_new_tokens: int = 100,
    ):
        sampling_params = {
            "temperature": 0.0001,
            "top_k": 10,
            "top_p": 0.9,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        }
        engine = AsyncLLMEngine(
            hf_model_identifier,
            ServerArgs(
                attention_backend=attention_backend, sampling_backend=sampling_backend
            ),
        )
        sp_outputs = [
            engine.generate(prompt, sampling_params=sampling_params)
            for prompt in prompts
        ]
        sp_outputs = sp_outputs[0]["text"]
        engine.shutdown()
        hf_outputs = clm_generate(hf_model_identifier, prompts, sampling_params)
        hf_outputs = hf_outputs[0]
        self.assertEqual(hf_outputs[:32], sp_outputs[:32])
        del engine

    @pytest.mark.gpu
    def test_llama_1b(self):
        model_identifiers = ["meta-llama/Llama-3.2-1B-Instruct"]
        prompts = ["Alan Turing is"]
        for model_id in model_identifiers:
            for attention_backend in self.attention_backends:
                for sampling_backend in self.sampling_backends:
                    print(f"Running {model_id} with {attention_backend}")
                    self.run_generation(
                        model_id, prompts, attention_backend, sampling_backend
                    )


if __name__ == "__main__":
    unittest.main()
