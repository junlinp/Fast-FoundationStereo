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
# make_onnx.py exports feature_runner.onnx and post_runner.onnx (separate, for two-engine TRT)
python scripts/make_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path output/480x864 \
  --height 480 \
  --width 864 \
  --valid_iters 4 \
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

#trtexec --onnx=output/480x864/feature_runner.onnx --saveEngine=output/480x864/feature_runner.engine --fp16  --useCudaGraph
#trtexec --onnx=output/480x864/post_runner.onnx --saveEngine=output/480x864/post_runner.engine --fp16  --useCudaGraph

python scripts/run_demo_tensorrt.py \
  --onnx_dir output/480x864/ \
  --left_file test_data/left.png \
  --right_file test_data/right.png \
  --intrinsic_file test_data/K.txt \
  --out_dir output/trt_inference_480x864/ \
  --remove_invisible 0 \
  --denoise_cloud 0 \
  --get_pc 0 \
  --zfar 100 