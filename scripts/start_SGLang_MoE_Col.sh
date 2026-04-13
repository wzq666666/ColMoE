CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --model-path /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct \
  --tp 1 \
  --port 30001 \
  --kt-method LLAMAFILE \
  --kt-weight-path /home/ecnu/disk/wzq/models/Qwen2-57B-A14B-Instruct-gguf \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 10 \
  --kt-max-deferred-experts-per-token 0 \
  --dtype bfloat16 \
  --max-total-tokens 10240 \
  --max-prefill-tokens 5120 \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph