# CLI Reference

**Usage**:

```console
$ scratchpad [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `benchmark`
* `chat`
* `serve`: Spin up the server
* `version`: Print the version

## `scratchpad benchmark`

**Usage**:

```console
$ scratchpad benchmark [OPTIONS] MODEL
```

**Arguments**:

* `MODEL`: [required]

**Options**:

* `--tasks TEXT`: [default: mmlu]
* `--url TEXT`: [default: http://localhost:8080/v1]
* `--num-fewshot INTEGER`: [default: 0]
* `--instruct-model / --no-instruct-model`: [default: no-instruct-model]
* `--help`: Show this message and exit.

## `scratchpad chat`

**Usage**:

```console
$ scratchpad chat [OPTIONS] MODEL
```

**Arguments**:

* `MODEL`: [required]

**Options**:

* `--backend TEXT`: [default: http://localhost:8080]
* `--help`: Show this message and exit.

## `scratchpad serve`

Spin up the server

**Usage**:

```console
$ scratchpad serve [OPTIONS] MODEL
```

**Arguments**:

* `MODEL`: [required]

**Options**:

* `--host TEXT`: [default: 0.0.0.0]
* `--port INTEGER`: [default: 3000]
* `--debug / --no-debug`: [default: no-debug]
* `--model-path TEXT`
* `--served-model-name TEXT`: [default: auto]
* `--trust-remote-code / --no-trust-remote-code`: [default: trust-remote-code]
* `--json-model-override-args TEXT`: [default: {}]
* `--is-embedding / --no-is-embedding`: [default: no-is-embedding]
* `--context-length INTEGER`: [default: 4096]
* `--skip-tokenizer-init / --no-skip-tokenizer-init`: [default: no-skip-tokenizer-init]
* `--tokenizer-path TEXT`: [default: auto]
* `--tokenizer-mode TEXT`: [default: auto]
* `--schedule-policy TEXT`: [default: lpm]
* `--random-seed INTEGER`
* `--stream-interval INTEGER`: [default: 1]
* `--chunked-prefill-size INTEGER`: [default: 8192]
* `--max-prefill-tokens INTEGER`: [default: 16384]
* `--max-running-requests INTEGER`
* `--max-total-tokens INTEGER`
* `--kv-cache-dtype TEXT`: [default: auto]
* `--schedule-conservativeness FLOAT`: [default: 1.0]
* `--load-format TEXT`: [default: auto]
* `--quantization TEXT`
* `--dtype TEXT`: [default: auto]
* `--dist-init-addr TEXT`
* `--dp-size INTEGER`: [default: 1]
* `--tp-size INTEGER`: [default: 1]
* `--nnodes INTEGER`: [default: 1]
* `--node-rank INTEGER`: [default: 0]
* `--load-balance-method TEXT`: [default: round_robin]
* `--tokenizer-port INTEGER`: [default: 30001]
* `--scheduler-port INTEGER`: [default: 30002]
* `--detokenizer-port INTEGER`: [default: 30003]
* `--nccl-ports TEXT`: [default: 30004]
* `--init-new-token-ratio FLOAT`: [default: 0.7]
* `--base-min-new-token-ratio FLOAT`: [default: 0.1]
* `--new-token-ratio-decay FLOAT`: [default: 0.001]
* `--num-continue-decode-steps INTEGER`: [default: 10]
* `--retract-decode-steps INTEGER`: [default: 20]
* `--mem-fraction-static FLOAT`: [default: 0.8]
* `--constrained-json-whitespace-pattern TEXT`
* `--lora-paths TEXT`
* `--max-loras-per-batch INTEGER`: [default: 1]
* `--attention-backend TEXT`
* `--sampling-backend TEXT`
* `--disable-flashinfer / --no-disable-flashinfer`: [default: no-disable-flashinfer]
* `--disable-flashinfer-sampling / --no-disable-flashinfer-sampling`: [default: no-disable-flashinfer-sampling]
* `--disable-radix-cache / --no-disable-radix-cache`: [default: no-disable-radix-cache]
* `--disable-regex-jump-forward / --no-disable-regex-jump-forward`: [default: no-disable-regex-jump-forward]
* `--disable-cuda-graph / --no-disable-cuda-graph`: [default: no-disable-cuda-graph]
* `--disable-cuda-graph-padding / --no-disable-cuda-graph-padding`: [default: no-disable-cuda-graph-padding]
* `--disable-disk-cache / --no-disable-disk-cache`: [default: no-disable-disk-cache]
* `--disable-custom-all-reduce / --no-disable-custom-all-reduce`: [default: no-disable-custom-all-reduce]
* `--disable-mla / --no-disable-mla`: [default: no-disable-mla]
* `--enable-mixed-chunk / --no-enable-mixed-chunk`: [default: no-enable-mixed-chunk]
* `--enable-torch-compile / --no-enable-torch-compile`: [default: no-enable-torch-compile]
* `--max-torch-compile-bs INTEGER`: [default: 32]
* `--torchao-config TEXT`
* `--enable-p2p-check / --no-enable-p2p-check`: [default: no-enable-p2p-check]
* `--flashinfer-workspace-size INTEGER`: [default: 402653184]
* `--triton-attention-reduce-in-fp32 / --no-triton-attention-reduce-in-fp32`: [default: no-triton-attention-reduce-in-fp32]
* `--log-requests / --no-log-requests`: [default: no-log-requests]
* `--show-time-cost / --no-show-time-cost`: [default: no-show-time-cost]
* `--help`: Show this message and exit.

## `scratchpad version`

Print the version

**Usage**:

```console
$ scratchpad version [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
