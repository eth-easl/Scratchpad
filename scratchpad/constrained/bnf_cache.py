"""Cache for the compressed finite state machine."""

from typing import Tuple

from transformers import AutoTokenizer

from scratchpad.constrained import (
    GrammarMatcher,
    GrammarMatcherInitContext,
    GrammarMatcherInitContextCache,
)

MAX_ROLLBACK_TOKENS = 10


class BNFCache:
    grammar_cache: GrammarMatcherInitContextCache

    def __init__(
        self,
        tokenizer_path,
        tokenizer_args_dict,
        skip_tokenizer_init=False,
        whitespace_patterns=None,
    ):
        # TODO(dark): how to deal with whitespace_patterns and skip_tokenizer_init
        if skip_tokenizer_init:
            return

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_args_dict)
        self.grammar_cache = GrammarMatcherInitContextCache(
            tokenizer_or_vocab=tokenizer
        )

    def get_context(self, key: Tuple[str, str]) -> GrammarMatcherInitContext:
        key_type, key_string = key
        if key_type == "json":
            return self.grammar_cache.get_init_context_for_json_schema(key_string)
        elif key_type == "regex":
            raise ValueError(f"regex hasn't been supported by xgrammar yet")
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

    def query(self, key: Tuple[str, str], vocab_size: int) -> GrammarMatcher:
        ctx = self.get_context(key)
        return GrammarMatcher(
            ctx, max_rollback_tokens=MAX_ROLLBACK_TOKENS, mask_vocab_size=vocab_size
        )
