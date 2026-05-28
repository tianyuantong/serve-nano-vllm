# Benchmark and Environment Notes

These commands capture the upstream baseline workflow. The current local machine
can complete repository setup and Python 3.12 virtualenv creation, but full
runtime installation needs a CUDA/Linux environment.

## Local Setup Attempt

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python --version
.venv/bin/pip install -e .
```

Observed local result:

- Python version: `Python 3.12.13`.
- `pip install -e .` reaches dependency resolution.
- Installation fails on macOS arm64 because `triton>=3.0.0` has no matching
  distribution for this platform.

This is consistent with the implementation importing CUDA-oriented packages and
calling CUDA APIs in `ModelRunner`.

## Expected CUDA/Linux Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the README model:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Run the upstream example:

```bash
python example.py
```

Run the upstream benchmark:

```bash
python bench.py
```

## Benchmark Shape in `bench.py`

- Model path: `~/huggingface/Qwen3-0.6B/`.
- Number of sequences: `256`.
- Input length: random between `100` and `1024` tokens.
- Output length: random between `100` and `1024` tokens.
- Warmup: one call to `llm.generate`.
- Metric: total configured output tokens divided by elapsed seconds.

## Next Baseline Tasks

- Run `pip install -e .` on a CUDA/Linux machine.
- Download `Qwen/Qwen3-0.6B`.
- Capture `python example.py` output.
- Capture `python bench.py` throughput.
- Record GPU model, driver, CUDA version, Python version, and package versions.
