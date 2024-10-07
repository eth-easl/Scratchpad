from transformers import AutoConfig
from tools.simulator.utils import (
    memory_matmul,
    flops_matmul,
    roofline_analyze,
    get_linear_layers,
)
from tools.simulator.hardware_params import hardware_params
import humanize


class RooflineSimulator:

    def __init__(
        self,
        model_name: str,
        hardware: str,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
    ) -> None:
        self.config = AutoConfig.from_pretrained(model_name)
        self.hw_name = hardware
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.hardware = hardware_params[hardware]
        self.analyze_results = {
            "prefill": {},
            "decode": {},
        }

    def _write_result(self, name, stage, k, v):
        if name not in self.analyze_results[stage]:
            self.analyze_results[stage][name] = {}
        if k not in self.analyze_results[stage][name]:
            self.analyze_results[stage][name][k] = None
        self.analyze_results[stage][name][k] = v

    def _analyze_linear_module(self, name, ic, oc, bsz, seq_len):
        """prefill first"""
        is_kv_proj = name in ["k_proj", "v_proj"]
        is_normal_proj = not is_kv_proj
        ops = flops_matmul(bsz, seq_len, ic, oc)
        weight_load, act_load = memory_matmul(
            bsz, ic, oc, seq_len, w_bit=self.w_bit, a_bit=self.a_bit
        )
        act_write = 0 if is_kv_proj else oc * bsz * seq_len * self.a_bit / 8
        kv_write = 0 if is_normal_proj else oc * bsz * seq_len * self.kv_bit / 8
        self._write_result(name, "prefill", "ops", ops)
        self._write_result(name, "prefill", "weight_load", weight_load)
        self._write_result(name, "prefill", "act_load", act_load)
        self._write_result(name, "prefill", "act_write", act_write)
        self._write_result(name, "prefill", "kv_write", kv_write)
        """ then decode """

    def _analyze(self, bsz, seq_len, tp=1):
        num_hidden_layers = self.config.num_hidden_layers
        linear_layers = get_linear_layers(
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_key_value_heads,
            self.config.num_attention_heads,
        )

        for name, (ic, oc) in linear_layers.items():
            self._analyze_linear_module(name, ic, oc, bsz, seq_len)

    @property
    def results(self):
        return self.analyze_results

    def display(self):
        for stage in ["prefill", "decode"]:
            for name, values in self.analyze_results[stage].items():
                print(f"Stage: {stage}, Module: {name}")
                for k, v in values.items():
                    print(f"{k}: {humanize.intword(v)}")
                print("\n")


if __name__ == "__main__":
    roofline = RooflineSimulator("meta-llama/Llama-2-7b-hf", "nvidia_A100")
    roofline._analyze(bsz=1, seq_len=1024)
    roofline.display()
