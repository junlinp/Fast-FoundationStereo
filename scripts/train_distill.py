#!/usr/bin/env python3
"""
Distillation Training Script for Stereo Matching
Teacher: FastFoundationStereo (16.9M params)
Student: DistilledStereo (3.5M params)
"""

import os
import sys
import time
import argparse
import logging

# Disable torch.compile for older GPUs (CUDA 6.1)
import torch
import cv2
import numpy as np
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')

from core.foundation_stereo import FastFoundationStereo
from core.student_model import DistilledStereo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_disparity(disp, target_h=None, target_w=None, max_disp=192):
    """Convert disparity to colorized visualization.
    disp: (H, W) tensor or (B, 1, H, W)
    target_h, target_w: target dimensions to resize to
    Returns: (H, W, 3) numpy array in RGB, resized if target dims provided
    """
    if isinstance(disp, torch.Tensor):
        disp = disp.detach().cpu().numpy()
    disp = disp.squeeze()
    
    # Normalize to [0, 1]
    disp_min = disp.min()
    disp_max = disp.max()
    if disp_max > disp_min:
        disp_norm = (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_norm = disp * 0
    
    # Apply colormap
    disp_uint8 = (disp_norm * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_TURBO)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    # Resize if target dimensions provided
    if target_h is not None and target_w is not None:
        colorized = cv2.resize(colorized, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return colorized


def log_disparity_to_wandb(student_disp, teacher_disp, left_img, step, max_disp=192):
    """Log student/teacher disparity comparison to wandb.
    """
    # Get numpy arrays
    if isinstance(student_disp, torch.Tensor):
        student_np = student_disp.detach().cpu().numpy()
    else:
        student_np = student_disp
    
    if isinstance(teacher_disp, torch.Tensor):
        teacher_np = teacher_disp.detach().cpu().numpy()
    else:
        teacher_np = teacher_disp
    
    if isinstance(left_img, torch.Tensor):
        left_np = left_img.detach().cpu().numpy()
    else:
        left_np = left_img
    left_np = left_np[0].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    left_np = (left_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    left_np = (left_np * 255).clip(0, 255).astype(np.uint8)
    
    # Create visualizations (resize to match left image at full res)
    h, w = left_np.shape[:2]
    student_vis = visualize_disparity(student_np, h, w, max_disp)
    teacher_vis = visualize_disparity(teacher_np, h, w, max_disp)
    
    # Stack side by side: left, teacher, student
    combined = np.zeros((h, w * 3 + 10 * 2, 3), dtype=np.uint8)
    combined[:, :w] = left_np
    combined[:, w + 10:w + 10 + w] = teacher_vis
    combined[:, w * 2 + 20:] = student_vis
    
    # Add labels (text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Left', (5, 20), font, 0.6, (255, 255, 255), 1)
    cv2.putText(combined, 'Teacher', (w + 15, 20), font, 0.6, (255, 255, 255), 1)
    cv2.putText(combined, 'Student', (w * 2 + 25, 20), font, 0.6, (255, 255, 255), 1)
    
    # Log to wandb
    wandb.log({
        "disparity_comparison": wandb.Image(combined, caption="Left | Teacher | Student"),
        "step": step
    })


class StereoDataset(torch.utils.data.Dataset):
    """
    Simple stereo dataset for distillation training.
    Supports KITTI-360 format (image_00/image_01 directories).
    """
    def __init__(self, data_dir, split='train', height=480, width=864, max_samples=1000):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        
        # Find stereo pairs
        self.left_images = []
        self.right_images = []
        
        # Check for KITTI-360 format (image_00/image_01 subdirs)
        for root, dirs, files in os.walk(data_dir):
            if 'image_00' in dirs and 'image_01' in dirs:
                left_dir = os.path.join(root, 'image_00', 'data_rect')
                right_dir = os.path.join(root, 'image_01', 'data_rect')
                if os.path.exists(left_dir) and os.path.exists(right_dir):
                    try:
                        left_files = sorted([f for f in os.listdir(left_dir) if f.endswith('.png')])
                        right_files = sorted([f for f in os.listdir(right_dir) if f.endswith('.png')])
                        # Sample to limit dataset size
                        step = max(1, len(left_files) // 200)
                        for f in left_files[::step]:
                            self.left_images.append(os.path.join(left_dir, f))
                        for f in right_files[::step]:
                            self.right_images.append(os.path.join(right_dir, f))
                    except Exception as e:
                        logger.warning(f"Error reading {left_dir}: {e}")
        
        # Sort and pair
        self.left_images.sort()
        self.right_images.sort()
        
        # Limit samples
        min_len = min(len(self.left_images), len(self.right_images))
        if max_samples:
            min_len = min(min_len, max_samples)
        self.left_images = self.left_images[:min_len]
        self.right_images = self.right_images[:min_len]
        
        logger.info(f'Loaded {len(self.left_images)} stereo pairs from {data_dir}')
    
    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, idx):
        # Load images
        left = cv2.imread(self.left_images[idx])
        right = cv2.imread(self.right_images[idx])
        
        if left is None or right is None:
            # Return blank images if loading fails
            left = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            right = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            # Resize
            left = cv2.resize(left, (self.width, self.height))
            right = cv2.resize(right, (self.width, self.height))
        
        # Convert to tensor
        left = torch.from_numpy(left.astype(np.float32) / 255.0).permute(2, 0, 1)
        right = torch.from_numpy(right.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        return left, right


class DistillLoss(nn.Module):
    """Distillation loss combining feature matching and output matching"""
    def __init__(self):
        super().__init__()
    
    def forward(self, student_out, teacher_out, left_image=None):
        """
        student_out: (B, 1, H, W) student disparity
        teacher_out: (B, 1, H, W) teacher disparity
        left_image: (B, 3, H, W) left image for edge-aware loss
        """
        losses = {}
        
        # 1. Smooth L1 loss (main distillation loss)
        teacher_out_resized = F.interpolate(teacher_out.detach(), size=student_out.shape[2:], mode='bilinear', align_corners=True)
        losses["smooth_l1"] = F.smooth_l1_loss(student_out, teacher_out_resized)
        
        # 2. Simple gradient matching
        if student_out.shape[-1] > 1:
            student_dx = student_out[:, :, :, 1:] - student_out[:, :, :, :-1]
            teacher_dx = teacher_out_resized.detach()[:, :, :, 1:] - teacher_out_resized.detach()[:, :, :, :-1]
            losses['gradient_x'] = F.smooth_l1_loss(student_dx, teacher_dx)
        
        if student_out.shape[-2] > 1:
            student_dy = student_out[:, :, 1:, :] - student_out[:, :, :-1, :]
            teacher_dy = teacher_out_resized.detach()[:, :, 1:, :] - teacher_out_resized.detach()[:, :, :-1, :]
            losses['gradient_y'] = F.smooth_l1_loss(student_dy, teacher_dy)
        
        # 3. Simple edge-aware loss (if image provided)
        if left_image is not None:
            # Resize left image to match disparity
            left_resized = F.interpolate(left_image, size=student_out.shape[-2:], mode='bilinear', align_corners=True)
            # Compute image gradients
            img_dx = torch.abs(left_resized[:, :, :, 1:] - left_resized[:, :, :, :-1]).mean(dim=1, keepdim=True)
            img_dy = torch.abs(left_resized[:, :, 1:, :] - left_resized[:, :, :-1, :]).mean(dim=1, keepdim=True)
            # Edge-aware: weight by image gradients
            if student_out.shape[-1] > 1:
                edge_loss_x = torch.abs(student_dx - teacher_dx) * (1.0 - img_dx[:, :, :, :student_dx.shape[-1]])
                losses['edge_x'] = edge_loss_x.mean()
            if student_out.shape[-2] > 1:
                edge_loss_y = torch.abs(student_dy - teacher_dy) * (1.0 - img_dy[:, :, :student_dy.shape[-2], :])
                losses['edge_y'] = edge_loss_y.mean()
        
        # Weighted sum
        total = (losses['smooth_l1'] * 1.0 + 
                 losses.get('gradient_x', 0.0) * 0.5 + 
                 losses.get('gradient_y', 0.0) * 0.5 +
                 losses.get('edge_x', 0.0) * 0.1 + 
                 losses.get('edge_y', 0.0) * 0.1)
        
        losses['total'] = total
        return losses


def load_teacher(args):
    """Load pretrained teacher model (frozen)"""
    logger.info(f'Loading teacher from {args.teacher_ckpt}')
    
    # Load teacher config
    class TeacherArgs:
        hidden_dims = [128, 128, 128]
        n_gru_layers = 3
        max_disp = 192
        corr_radius = 4
        corr_levels = 2
        vit_size = 'vits'
        cv_group = 8
        volume_dim = 28
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    teacher_args = TeacherArgs()
    
    # Load weights
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    
    # Check if checkpoint is already a model object
    if isinstance(ckpt, torch.nn.Module):
        teacher = ckpt
    elif isinstance(ckpt, dict):
        teacher = FastFoundationStereo(teacher_args)
        if "model" in ckpt:
            teacher.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            teacher.load_state_dict(ckpt["state_dict"])
        else:
            teacher.load_state_dict(ckpt)
    else:
        teacher = FastFoundationStereo(teacher_args)
        teacher.load_state_dict(ckpt)
    
    # Freeze
    for param in teacher.parameters():
        param.requires_grad = False
    
    teacher.eval()
    return teacher


def load_student(args):
    """Load student model"""
    logger.info('Creating student model')
    
    class StudentArgs:
        hidden_dims = [128]
        max_disp = 192
        corr_radius = 2
        corr_levels = 1
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    student_args = StudentArgs()
    student = DistilledStereo(student_args)
    
    # Load pretrained if available
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        logger.info(f'Loading student from {args.student_ckpt}')
        ckpt = torch.load(args.student_ckpt, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                student.load_state_dict(ckpt['model'])
            elif 'state_dict' in ckpt:
                student.load_state_dict(ckpt['state_dict'])
            else:
                student.load_state_dict(ckpt)
        else:
            student.load_state_dict(ckpt)
    
    return student


def train_epoch(student, teacher, train_loader, optimizer, criterion, device, epoch, log_interval=10):
    """Train one epoch"""
    student.train()
    teacher.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (left, right) in enumerate(train_loader):
        left = left.to(device)
        right = right.to(device)
        
        # Forward pass with teacher (frozen)
        with torch.no_grad():
            teacher_out = teacher(left, right)[0]
        
        # Forward pass with student
        student_out = student(left, right)
        
        # Compute loss
        losses = criterion(student_out, teacher_out, left)
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f'Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, '
                       f'Loss: {avg_loss:.4f} '
                       f'(S1: {losses["smooth_l1"]:.4f})')
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx + 1,
                "loss": avg_loss,
                "loss_smooth_l1": losses["smooth_l1"].item(),
                "lr": optimizer.param_groups[0]['lr'],
            })
            
            # Log disparity visualization every 50 batches
            if (batch_idx + 1) % 50 == 0:
                log_disparity_to_wandb(student_out, teacher_out, left, epoch * 1000 + batch_idx)
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Distillation Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Teacher checkpoint path')
    parser.add_argument('--student_ckpt', type=str, default=None, help='Student checkpoint path (optional)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--height', type=int, default=480, help='Image height')
    parser.add_argument('--width', type=int, default=864, help='Image width')
    parser.add_argument('--max_samples', type=int, default=500, help='Max training samples per sequence')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load teacher
    teacher = load_teacher(args).to(device)
    logger.info(f'Teacher params: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M')
    
    # Load student
    student = load_student(args).to(device)
    logger.info(f'Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M')
    
    # Create dataset
    train_dataset = StereoDataset(args.data_dir, height=args.height, width=args.width, max_samples=args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project or "distill-stereo",
            name=args.wandb_name or f"student-{int(time.time())}",
            resume=args.resume is not None,
            config={
                "teacher_params": sum(p.numel() for p in teacher.parameters()) / 1e6,
                "student_params": sum(p.numel() for p in student.parameters()) / 1e6,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "height": args.height,
                "width": args.width,
                "max_samples": args.max_samples,
            }
        )
    
    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    criterion = DistillLoss()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    start_epoch = 1
    best_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        avg_loss = train_epoch(student, teacher, train_loader, optimizer, criterion, device, epoch, args.log_interval)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        logger.info(f'Epoch {epoch} complete, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
        })
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, 'student_best.pth')
            torch.save({
                'epoch': epoch,
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f'Saved best checkpoint: {save_path}')
        
        # Save latest
        save_path = os.path.join(args.save_dir, 'student_latest.pth')
        torch.save({
            'epoch': epoch,
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        wandb.save(save_path)
    
    logger.info('Training complete!')
    
    # Finish wandb
    wandb.finish()


if __name__ == '__main__':
    main()
