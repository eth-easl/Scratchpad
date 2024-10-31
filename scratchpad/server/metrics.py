import numpy as np
from typing import List, Dict, Union
import prometheus_client
from typing import Counter as CollectionsCounter
from scratchpad.server.metric_types import StatLoggerBase, SupportsMetricsInfo
from scratchpad.scheduler.stats import Stats


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


class Metrics:
    labelname_finish_reason = "finished_reason"
    labelname_waiting_lora_adapters = "waiting_lora_adapters"
    labelname_running_lora_adapters = "running_lora_adapters"
    labelname_max_lora = "max_lora"
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, labelnames: List[str], max_model_len: int):
        self.gauge_scheduler_running = self._gauge_cls(
            name="scratchpad:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_scheduler_waiting = self._gauge_cls(
            name="scratchpad:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_mempool_usage = self._gauge_cls(
            name="scratchpad:mempool_usage",
            documentation="Memory pool usage.",
            labelnames=labelnames,
            multiprocess_mode="mostrecent",
        )
        self.gauge_mempool_usage_pct = self._gauge_cls(
            name="scratchpad:mempool_usage_percent",
            documentation="Memory pool usage (pct).",
            labelnames=labelnames,
            multiprocess_mode="mostrecent",
        )
        self.gauge_avg_prompt_throughput = self._gauge_cls(
            name="scratchpad:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gauge_avg_generation_throughput = self._gauge_cls(
            name="scratchpad:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )


def local_interval_elapsed(now: float, last_log: float, local_interval: float) -> bool:
    elapsed_time = now - last_log
    return elapsed_time > local_interval


def get_throughput(tracked_stats: List[int], now: float, last_log: float) -> float:
    return float(np.sum(tracked_stats) / (now - last_log))


class PrometheusStatLogger(StatLoggerBase):
    """PrometheusStatLogger is used LLMEngine to log to Promethus."""

    _metrics_cls = Metrics
    _gauge_cls = prometheus_client.Gauge

    def __init__(
        self, local_interval: float, labels: Dict[str, str], max_model_len: int
    ) -> None:
        super().__init__(local_interval)
        # Prometheus metrics
        self.labels = labels
        self.metrics = self._metrics_cls(
            labelnames=list(labels.keys()), max_model_len=max_model_len
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(
        self, counter, data: CollectionsCounter, label_key: str
    ) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def _log_gauge_string(self, gauge, data: Dict[str, str]) -> None:
        gauge.labels(**data).set(1)

    def _log_prometheus(self, stats: Stats) -> None:
        self._log_gauge(
            self.metrics.gauge_avg_generation_throughput, stats.generation_throughput
        )
        self._log_gauge(
            self.metrics.gauge_scheduler_running,
            stats.running_requests,
        )
        self._log_gauge(
            self.metrics.gauge_scheduler_waiting,
            stats.queued_requests,
        )
        self._log_gauge(
            self.metrics.gauge_mempool_usage_pct,
            100 * stats.token_usage,
        )
        self._log_gauge(
            self.metrics.gauge_mempool_usage,
            stats.used_token_pool,
        )

    def _log_prometheus_interval(
        self, prompt_throughput: float, generation_throughput: float
    ) -> None:

        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on
        # the vLLM side. Moving forward, we should use counters like
        # counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the
        # grafana/prometheus side. See
        # https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        self.metrics.gauge_avg_prompt_throughput.labels(**self.labels).set(
            prompt_throughput
        )
        self.metrics.gauge_avg_generation_throughput.labels(**self.labels).set(
            generation_throughput
        )

    def log(self, stats: Stats):
        """Logs to prometheus and tracked stats every iteration."""
        # Log to prometheus.
        self._log_prometheus(stats)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        # Info type metrics are syntactic sugar for a gauge permanently set to 1
        # Since prometheus multiprocessing mode does not support Info, emulate
        # info here with a gauge.
        if type == "cache_config":
            metrics_info = obj.metrics_info()
            info_gauge = self._gauge_cls(
                name="scratchpad:cache_config_info",
                documentation="Information of the LLMEngine CacheConfig",
                labelnames=metrics_info.keys(),
                multiprocess_mode="mostrecent",
            )
            info_gauge.labels(**metrics_info).set(1)
