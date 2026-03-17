import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio.v2 as imageio, logging, yaml, time
import numpy as np
from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
from core.foundation_stereo import TrtSingleRunner
import cv2


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--engine', default=f'{code_dir}/../output/480x864/foundation_stereo.engine', type=str)
  parser.add_argument('--plugin_lib', default=f'{code_dir}/../plugins/gwc_plugin_cpp/build/libGwcVolumePlugin.so', type=str,
                      help='Path to GwcVolume TensorRT plugin .so (for deserialization)')
  parser.add_argument('--onnx_dir', default=f'{code_dir}/../output/480x864', type=str,
                      help='Directory containing onnx.yaml config (image_size, etc.)')
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str,
                      help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/trt_single_inference_480x864', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=1, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30,
                      help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03,
                      help='radius to use for outlier removal')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")
  parser.add_argument('--benchmark', action='store_true',
                      help='run speed benchmark (warmup + iterations, no GUI)')
  parser.add_argument('--benchmark_warmup', type=int, default=15,
                      help='warmup iterations for benchmark')
  parser.add_argument('--benchmark_total', type=int, default=30,
                      help='total iterations for benchmark')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  if not args.benchmark:
    os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

  # Load config from onnx.yaml to get image_size and model args (max_disp, etc.)
  cfg_path = f'{args.onnx_dir}/onnx.yaml'
  if not os.path.isfile(cfg_path):
    cfg_path = f'{os.path.dirname(args.onnx_dir)}/onnx.yaml'
  with open(cfg_path, 'r') as f:
    cfg:dict = yaml.safe_load(f)
  if 'image_size' not in cfg:
    name = os.path.basename(args.onnx_dir.rstrip('/'))
    if 'x' in name:
      parts = name.split('x')
      if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
        cfg['image_size'] = [int(parts[0]), int(parts[1])]
  if 'image_size' not in cfg:
    cfg['image_size'] = [480, 864]
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  do_benchmark = cfg.get('benchmark', False)
  benchmark_warmup = cfg.get('benchmark_warmup', 15)
  benchmark_total = cfg.get('benchmark_total', 30)
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")

  model = TrtSingleRunner(
      args,
      engine_path=args.engine,
      plugin_lib=args.plugin_lib,
  )

  if do_benchmark and not os.path.isfile(args.left_file):
    H_cfg, W_cfg = args.image_size[0], args.image_size[1]
    img0 = np.random.randint(0, 256, (H_cfg, W_cfg, 3), dtype=np.uint8)
    img1 = np.random.randint(0, 256, (H_cfg, W_cfg, 3), dtype=np.uint8)
  else:
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
  if len(img0.shape)==2:
    img0 = np.tile(img0[...,None], (1,1,3))
    img1 = np.tile(img1[...,None], (1,1,3))
  img0 = img0[...,:3]
  img1 = img1[...,:3]
  H,W = img0.shape[:2]

  fx = args.image_size[1] / img0.shape[1]
  fy = args.image_size[0] / img0.shape[0]
  if fx != 1 or fy != 1:
    logging.info(f">>>>>>>>>>>>>>>WARNING: resizing image to {args.image_size}, fx: {fx}, fy: {fy}, this is not recommended. It's best to make tensorrt engine with the same image size as the input image.")
  img0 = cv2.resize(img0, fx=fx, fy=fy, dsize=None)
  img1 = cv2.resize(img1, fx=fx, fy=fy, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f"img0: {img0.shape}")
  if not do_benchmark:
    imageio.imwrite(f'{args.out_dir}/left.png', img0)
    imageio.imwrite(f'{args.out_dir}/right.png', img1)

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)

  if do_benchmark:
    logging.info(f"Benchmark (single engine): image {H}x{W}, warmup={benchmark_warmup}, total={benchmark_total}")
    times = []
    for i in range(benchmark_total):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      disp = model.forward(img0, img1)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - t0
      times.append(elapsed)
      logging.info(f"Iter {i:2d}: {elapsed*1000:.1f} ms {'(warmup)' if i < benchmark_warmup else ''}")
    measure_times = times[benchmark_warmup:]
    avg_ms = np.mean(measure_times) * 1000
    std_ms = np.std(measure_times) * 1000
    logging.info(f"TensorRT single-engine speed average (after warmup): {avg_ms:.1f} ± {std_ms:.1f} ms over {len(measure_times)} iters")
    sys.exit(0)

  logging.info(f"Start forward (single engine), 1st time run can be slow due to CUDA graph capture")
  disp = model.forward(img0, img1)
  logging.info("forward done")
  disp = disp.data.cpu().numpy().reshape(H,W).clip(0, None) * 1/fx

  cmap = None
  min_val = None
  max_val = None
  vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)
  s = 1280/vis.shape[1]
  resized_vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= np.array([fx, fy])
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    id = np.asarray(pcd.points)[:,2].argmin()
    ctr.set_lookat(np.asarray(pcd.points)[id])
    ctr.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()

