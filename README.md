# EagleMoE

## Layout

- `src/`: inference core and hook runtime
- `configs/`: runtime config templates for `our`, `smoe`, `static`, and `hybri` (see [`configs/README.md`](configs/README.md))
- `examples/`: runnable examples and client requests
- `experiments/`: kept empty for paper reproduction notes or future experiments
- `utils/`: shared helpers for examples and data loading
- `docs/`: setup and usage notes
- `scripts/`: launch scripts for SGLang and hook-based serving

## Supported Inference Paths

The public runtime surface keeps only four routing paths:

- `our`: load-aware token routing used by `configs/moe_hook_our.yaml`
- `smoe`: low-score token routing used by `configs/moe_hook_smoe.yaml`
- `static`: no reroute path used by `configs/moe_hook_static.yaml`
- `hybri`: hybrid runtime scheduling used by `configs/moe_hook_hybri.yaml`

## Quick Start

1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

To use the bundled KTransformers runtime, install `kt-kernel` as well:

```bash
cd /home/ecnu/disk/wzq/ktransformers/kt-kernel
pip install .
```

2. Set your local model paths:

```bash
export MODEL_PATH=/path/to/your/model
export GGUF_MODEL_PATH=/path/to/your/model-gguf
export MOE_HOOK_CONFIG=$(pwd)/configs/moe_hook_our.yaml
export CUDA_VISIBLE_DEVICES=xxx
```

3. Start the hook-based server:

```bash
bash scripts/start_SGLang_MoE_Col_with_Hook.sh
```

4. Send a request with the example client:

```bash
python examples/send_infer_request.py --data-dir /path/to/your/dataset
```

For environment setup details, see `docs/ENVIRONMENT.md`.

## Script Guide

The `scripts/` directory contains the main launch entry points:

- `scripts/start_SGLang_MoE_Col_with_Hook.sh`: full hook-based launcher. It sets runtime defaults such as `CUDA_VISIBLE_DEVICES`, `MOE_HOOK_CONFIG`, `MEM_FRACTION_STATIC`, and `KT_NUM_GPU_EXPERTS`, then starts the hook-enabled SGLang server.
- `scripts/run_with_hooks.sh`: thin wrapper that prepares `PYTHONPATH`, enables the hook config, and calls `run_with_hooks.py` with the remaining arguments.
- `scripts/start_SGLang_MoE_Col.sh`: plain SGLang + KTransformers launch without the MoE hook layer. Use this if you want the base heterogeneous inference path without the hook-based routing logic.
- `scripts/run_sglang_in_backend.sh`: convenience wrapper that starts `start_SGLang_MoE_Col_with_Hook.sh` in a detached `screen` session named `sglang`.

## Notes and runtime caveats

- `--kt-num-gpu-experts`: number of GPU-resident experts per MoE layer used by the native/ktransformers runtime. This value must match the expectations in your `MOE_HOOK_CONFIG` YAML (e.g., if the config assumes 10 GPU experts per layer, set `--kt-num-gpu-experts 10`). Larger values increase GPU memory usage but may improve throughput when many tokens target GPU experts.
- `mem-fraction-static`: reserves a fraction of GPU memory for static allocations. Use a value like `0.7-0.9` to avoid OOMs when model and expert weights are loaded. If you see OOMs, lower this value or reduce `--kt-num-gpu-experts`.
- `--disable-cuda-graph` / cuda graph flags: dynamic scheduling (the `hybri` strategy and other runtime re-routing) is generally incompatible with CUDA graph capture. The start script disables CUDA graph by default for dynamic scheduling — keep it disabled when using dynamic reroute strategies.
- `CUDA_VISIBLE_DEVICES` mapping: the launcher sets `CUDA_VISIBLE_DEVICES` as a default when unset; remember this remaps logical `cuda:0` to the selected physical GPU. Verify your process sees the intended GPU with:

```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi -i $(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
```

- Debugging tips: if your process is killed with exit code 137 or you observe OOMs, check `dmesg` for OOM killer messages and lower `--kt-num-gpu-experts` or `mem-fraction-static` accordingly. Also confirm that any cuda-graph batch-size flags (`--cuda-graph-bs`, `--cuda-graph-max-bs`) are compatible with your dynamic scheduling setup.

