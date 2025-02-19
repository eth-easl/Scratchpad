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


class BaseGrammarBackend:
    def __init__(self):
        self.executor = ThreadPoolExecutor()
        self.cache = {}
        self.cache_lock = Lock()

    def init_value(self, key: Tuple[str, str]) -> BaseGrammarObject:
        with self.cache_lock:
            if key in self.cache:
                cache_hit = True
                entry = self.cache[key]
            else:
                cache_hit = False
                entry = CacheEntry(None, Event())
                self.cache[key] = entry

        if cache_hit:
            entry.event.wait()
        else:
            entry.value = self.init_value_impl(key)
            entry.event.set()
        return entry.value.copy() if entry.value else None

    def init_value_impl(self, key: Tuple[str, str]) -> BaseGrammarObject:
        raise NotImplementedError()

    def get_cached_value(self, key: Tuple[str, str]) -> Optional[BaseGrammarObject]:
        with self.cache_lock:
            entry = self.cache.get(key)
            if not entry or not entry.event.is_set():
                return None
            val = self.cache[key].value
            return val.copy() if val else None

    def get_future_value(self, key: Tuple[str, str]) -> Future:
        return self.executor.submit(self.init_value, key)

    def reset(self):
        with self.cache_lock:
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
