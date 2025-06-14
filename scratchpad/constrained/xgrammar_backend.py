from typing import List, Tuple
import torch
from xgrammar import (
    CompiledGrammar,
    Grammar,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
)

from .base_backend import BaseGrammarObject, BaseGrammarBackend
from scratchpad.utils import logger


MAX_ROLLBACK_TOKENS = 200


class XGrammarGrammar(BaseGrammarObject):
    def __init__(
        self, matcher: GrammarMatcher, vocab_size: int, ctx: CompiledGrammar
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.finished = False

    def accept_token(self, token: int):
        assert self.matcher.accept_token(token)

    def try_jump_forward(self, tokenizer) -> Tuple[List[int], str]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        apply_token_bitmask_inplace(logits, vocab_mask)

    def copy(self):
        matcher = GrammarMatcher(self.ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, self.ctx)


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        tokenizer_info = TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size

    def init_value_impl(self, key: Tuple[str, str]) -> XGrammarGrammar:

        key_type, key_string = key
        if key_type == "json":
            try:
                if key_string == "$$ANY$$":
                    ctx = self.grammar_compiler.compile_builtin_json_grammar()
                else:
                    ctx = self.grammar_compiler.compile_json_schema(schema=key_string)
            except RuntimeError as e:
                logger.warning(
                    f"Skip invalid json_schema: json_schema={key_string}, {e=}"
                )
                return None
        elif key_type == "ebnf":
            try:
                ctx = self.grammar_compiler.compile_grammar(key_string)
            except RuntimeError as e:
                logger.warning(f"Skip invalid ebnf: ebnf={key_string}, {e=}")
                return None
        elif key_type == "regex":
            try:
                ctx = self.grammar_compiler.compile_grammar(
                    Grammar.from_regex(key_string)
                )
            except RuntimeError as e:
                logger.warning(f"Skip invalid regex: regex={key_string}, {e=}")
                return None
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        matcher = GrammarMatcher(ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS)
        return XGrammarGrammar(matcher, self.vocab_size, ctx)

    def reset(self):
        if self.grammar_compiler:
            self.grammar_compiler.clear_cache()
