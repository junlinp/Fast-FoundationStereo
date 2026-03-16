# 480x864 full-resolution, low_memory=1 to reduce intermediate tensor sizes
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
  --max_disp 192 \
  --low_memory 1
