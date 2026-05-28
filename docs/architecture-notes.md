# nano-vLLM Architecture Notes

These notes summarize the upstream request lifecycle before this fork changes
serving or scheduling behavior.

## Entry Points

- `example.py` loads a local Qwen3 model, creates `LLM`, applies a chat template,
  and calls `LLM.generate`.
- `bench.py` generates random token-id prompts, warms up the engine, and measures
  output-token throughput.
- `nanovllm/llm.py` exposes `LLM` as a thin subclass of `LLMEngine`.

## Request Lifecycle

1. `LLMEngine.generate` receives prompts and sampling params.
2. `LLMEngine.add_request` tokenizes string prompts and creates a `Sequence`.
3. `Scheduler.add` appends the sequence to the waiting queue.
4. `LLMEngine.step` calls `Scheduler.schedule`.
5. `ModelRunner.run` prepares either a prefill batch or decode batch.
6. `Scheduler.postprocess` updates KV cache metadata, appends generated tokens,
   and finishes sequences on EOS or `max_tokens`.
7. `LLMEngine.generate` decodes finished token ids back to text.

## Request Object

`nanovllm/engine/sequence.py` defines `Sequence`. Important fields:

- `seq_id`: monotonically increasing request id.
- `status`: `WAITING`, `RUNNING`, or `FINISHED`.
- `token_ids`: prompt plus generated tokens.
- `num_prompt_tokens`: original prompt length.
- `num_cached_tokens`: prefix/KV-cache progress already covered.
- `num_scheduled_tokens`: tokens scheduled for the current step.
- `is_prefill`: whether the sequence is currently being prefetched.
- `block_table`: KV-cache block ids used by the sequence.
- `temperature`, `max_tokens`, `ignore_eos`: sampling parameters copied from
  `SamplingParams`.

## Scheduler

`nanovllm/engine/scheduler.py` owns two deques:

- `waiting`: sequences that still need prefill work.
- `running`: sequences ready for decode.

Scheduling prioritizes prefill first. If any waiting sequence can be scheduled,
`schedule` returns a prefill batch. If no prefill work is scheduled, decode
selects running sequences and schedules one token for each selected sequence.

Prefill can be chunked when the prompt does not fit into the remaining
`max_num_batched_tokens` budget, but only for the first scheduled sequence in
the batch. Decode may preempt running sequences when there is not enough KV
cache room to append another token.

## Prefill and Decode

`nanovllm/engine/model_runner.py` separates batch preparation:

- `prepare_prefill` builds token ids, positions, cumulative sequence lengths,
  slot mappings, and optional prefix-cache block tables.
- `prepare_decode` builds one-token inputs, positions, context lengths, slot
  mappings, and block tables.

`nanovllm/layers/attention.py` uses the context to choose the attention path:

- prefill uses `flash_attn_varlen_func`.
- decode uses `flash_attn_with_kvcache`.

CUDA graph replay is used for decode when eager mode is disabled and the batch
size fits the captured graph set.

## KV Cache

`nanovllm/engine/block_manager.py` tracks KV-cache blocks in host-side metadata.

- `BlockManager.can_allocate` checks prefix-cache hits and free-block capacity.
- `BlockManager.allocate` attaches cached and newly allocated blocks to a
  sequence.
- `BlockManager.can_append` and `may_append` reserve a new block when decode
  crosses a block boundary.
- `BlockManager.deallocate` decrements reference counts and returns unused
  blocks to the free queue.
- `BlockManager.hash_blocks` hashes completed blocks with `xxhash` so later
  requests can reuse matching prefixes.

`ModelRunner.allocate_kv_cache` creates the device-side KV-cache tensor after
measuring CUDA memory and calculating the number of blocks that fit inside
`gpu_memory_utilization`.

## Benchmark Path

`bench.py` uses token-id prompts rather than text prompts. It creates 256 random
requests, randomizes input and output lengths, runs one warmup generation, then
measures total output tokens divided by elapsed wall time.
