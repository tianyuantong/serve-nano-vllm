import argparse
import atexit
import gc

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


def make_llm(model: str):
    return LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=512,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enable_prefix_cache=False,
    )


def close_llm(llm):
    atexit.unregister(llm.exit)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main(model: str):
    prompt = [100 + i % 1000 for i in range(256)]
    params = SamplingParams(
        temperature=0,
        max_tokens=3,
        ignore_eos=True,
    )

    baseline_llm = make_llm(model)
    baseline = baseline_llm.generate(
        [prompt],
        params,
        use_tqdm=False,
    )[0]["token_ids"]
    assert len(baseline) == 3, baseline
    close_llm(baseline_llm)

    verify_llm = make_llm(model)
    verify_llm.add_request(prompt, params)
    outputs, scheduled_tokens = verify_llm.step()
    assert outputs == []
    assert scheduled_tokens == len(prompt)

    seq = verify_llm.scheduler.running[0]
    manager = verify_llm.scheduler.block_manager
    runner = verify_llm.model_runner

    assert seq.last_token == baseline[0]
    assert seq.num_cached_tokens == len(prompt)
    assert len(seq.block_table) == 1

    block_table_before = list(seq.block_table)
    used_before = len(manager.used_block_ids)
    free_before = len(manager.free_block_ids)

    manager.may_append(seq)
    assert len(seq.block_table) == len(block_table_before) + 1
    reserved_block = seq.block_table[-1]

    query = [seq.last_token, baseline[1]]
    start = seq.num_cached_tokens
    input_ids = torch.tensor(query, dtype=torch.int64, device="cuda")
    positions = torch.tensor(
        [start, start + 1],
        dtype=torch.int64,
        device="cuda",
    )
    slot_mapping = torch.tensor(
        [
            reserved_block * runner.block_size,
            reserved_block * runner.block_size + 1,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_q = torch.tensor(
        [0, 2],
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_k = torch.tensor(
        [0, start + 2],
        dtype=torch.int32,
        device="cuda",
    )
    block_tables = torch.tensor(
        [seq.block_table],
        dtype=torch.int32,
        device="cuda",
    )

    try:
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            2,
            start + 2,
            slot_mapping,
            None,
            block_tables,
        )
        try:
            with torch.inference_mode():
                hidden_states = runner.model(input_ids, positions)
                # Attention needs prefill geometry; the LM head must retain both rows.
                get_context().is_prefill = False
                logits = runner.model.compute_logits(hidden_states)
                predicted = logits.argmax(dim=-1).tolist()
        finally:
            reset_context()
    finally:
        removed_block = seq.block_table.pop()
        assert removed_block == reserved_block
        manager.blocks[reserved_block].ref_count -= 1
        manager._deallocate_block(reserved_block)

    assert logits.shape[0] == 2
    assert predicted == baseline[1:3], (predicted, baseline)
    assert seq.block_table == block_table_before
    assert len(manager.used_block_ids) == used_before
    assert len(manager.free_block_ids) == free_before

    print(f"BASELINE_TOKENS={baseline}")
    print(f"QUERY={query}")
    print(f"LOGITS_ROWS={logits.shape[0]}")
    print(f"VERIFY_ARGMAX={predicted}")
    print(f"EXPECTED={baseline[1:3]}")
    print(f"BLOCKS={used_before}->{used_before + 1}->{len(manager.used_block_ids)}")
    print("VERIFY_K1_GATE0_DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    main(args.model)
