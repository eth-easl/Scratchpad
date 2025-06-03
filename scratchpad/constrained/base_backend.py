from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Optional, Tuple

from scratchpad.server.args import ServerArgs


@dataclass
class CacheEntry:
    value: Any
    event: Event


class BaseGrammarObject:
    pass


INVALID_GRAMMAR_OBJ: BaseGrammarObject = BaseGrammarObject()


class BaseGrammarBackend:
    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache: Dict[Tuple[str, str], CacheEntry] = {}

    def _not_supported(self, key_type: str, key_string: str) -> None:
        logger.warning(f"Skip unsupported {key_type=}, {key_string=}")

    def dispatch_fallback(
        self, key_type: str, key_string: str
    ) -> Optional[BaseGrammarObject]:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    def dispatch_json(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("json", key_string)

    def dispatch_regex(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("regex", key_string)

    def dispatch_ebnf(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> Optional[BaseGrammarObject]:
        return self._not_supported("structural_tag", key_string)

    def _init_value_dispatch(self, key: Tuple[str, str]) -> Optional[BaseGrammarObject]:
        key_type, key_string = key
        if key_type == "json":
            return self.dispatch_json(key_string)
        elif key_type == "regex":
            return self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            return self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            return self.dispatch_structural_tag(key_string)
        elif key_type == "structural_pattern":
            return self.dispatch_structural_pattern(key_string)
        else:
            return self.dispatch_fallback(key_type, key_string)

    def get_cached_or_future_value(
        self, key: Tuple[str, str]
    ) -> Optional[BaseGrammarObject]:
        value = self.cache.get(key)
        if value:
            return value.copy(), True
        value = self.executor.submit(self._init_value_dispatch, key)
        return value, False

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value

    def reset(self):
        self.cache.clear()


def create_grammar_backend(server_args: ServerArgs, tokenizer, vocab_size):
    if server_args.grammar_backend == "outlines":
        from .outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
            allow_jump_forward=not server_args.disable_jump_forward,
        )
    elif server_args.grammar_backend == "xgrammar":
        from .xgrammar_backend import XGrammarGrammarBackend

        grammar_backend = XGrammarGrammarBackend(tokenizer, vocab_size=vocab_size)
    else:
        raise ValueError(f"Invalid grammar backend: {server_args.grammar_backend}")

    return grammar_backend
