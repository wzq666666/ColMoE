python send_infer_request.py \
  --port 30001 \
  --data-dir /home/ecnu/disk/wzq/moe-inference/data/gsm8k \
  --branch train \
  --data-idx 32 \
  --max-input-tokens 5120 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --use-requests \