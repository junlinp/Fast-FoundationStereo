# 480 / 2 -> 240. closest 32 div is 256
# 848 / 2 -> 424 closest  32 div is 448
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/256x448 \
  --height 256 \
  --width 448 \
  --valid_iters 4 \
  --max_disp 192

# 480x848 input, padded to 480x864 (nearest multiple of 32)
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
  --max_disp 192

# Merge feature_runner + post_runner into a single ONNX with a GWC plugin node.
# This enables building a single TRT engine (make ffs_single in tinynav/models).
python scripts/merge_onnx.py \
  --feature_onnx output/480x864/feature_runner.onnx \
  --post_onnx    output/480x864/post_runner.onnx \
  --output_onnx  output/480x864/foundation_stereo.onnx \
  --maxdisp 48 --num_groups 8

# 640 / 2 -> 320
# 544 / 2 -> 272 need to 288
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/320x288 \
  --height 320 \
  --width 288 \
  --valid_iters 16 \
  --max_disp 192