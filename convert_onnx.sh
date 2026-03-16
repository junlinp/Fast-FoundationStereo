# 480 / 2 -> 240. closest 32 div is 256
# 848 / 2 -> 424 closest  32 div is 448
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/256x448 \
  --height 256 \
  --width 448 \
  --valid_iters 16 \
  --max_disp 192

# 640 / 2 -> 320
# 544 / 2 -> 272 need to 288
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/320x288 \
  --height 320 \
  --width 288 \
  --valid_iters 16 \
  --max_disp 192

# 480x864 full-resolution, low_memory=1 to reduce intermediate tensor sizes
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
  --max_disp 192 \
  --low_memory 1
