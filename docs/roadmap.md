# serve-nano-vllm Roadmap

This fork is a learning and serving benchmark project built on top of
`GeeeekExplorer/nano-vllm`. The first milestone is to preserve the upstream
implementation, understand the request lifecycle, and add reproducible notes
before making serving or scheduler changes.

## Phase 0: Bootstrap

- Fork upstream into `lanandtianforever/serve-nano-vllm`.
- Clone the fork locally and add upstream as `GeeeekExplorer/nano-vllm`.
- Create the working branch `feat/serving-benchmark`.
- Verify the package install path and document environment blockers.
- Map the core code paths for requests, scheduling, model execution, and KV
  cache management.

## Phase 1: Architecture Notes

- Trace `LLM.generate` from request submission to token generation.
- Identify the request object and sequence state machine.
- Document how prefill and decode batches are scheduled.
- Document how KV cache blocks are allocated, reused, hashed, and released.
- Document how `bench.py` measures throughput.

## Phase 2: Serving Benchmark Harness

- Add a minimal reproducible benchmark command set.
- Add a small serving-oriented workload shape: request count, prompt length,
  output length, concurrency, and warmup policy.
- Capture baseline latency and throughput on a CUDA/Linux environment.
- Keep benchmark scripts separate from scheduler changes.

## Phase 3: Experiments

- Compare default scheduling behavior against serving-style workloads.
- Measure prefill and decode throughput separately.
- Add metrics that explain queueing, preemption, cache pressure, and generated
  tokens per second.
- Only change scheduler behavior after the baseline is reproducible.

## Current Local Status

- Fork created: `https://github.com/lanandtianforever/serve-nano-vllm`.
- Local branch created: `feat/serving-benchmark`.
- Python 3.12 virtual environment created at `.venv`.
- Full editable install is blocked on macOS arm64 because `triton>=3.0.0` has
  no matching distribution for this platform. The upstream runtime expects a
  CUDA/Linux environment and also imports `flash-attn`.
