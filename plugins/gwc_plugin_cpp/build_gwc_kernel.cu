#include <cuda_fp16.h>

// Note: blockDim.x/y are expected to be 16x16 (see launchBuildGwc).

extern "C" __global__ void build_gwc_f32(
    const float* __restrict__ ref, const float* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += ref[base + w] * ((tw >= 0) ? tgt[base + tw] : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}

extern "C" __global__ void build_gwc_f16(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    int w   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y * blockDim.y + threadIdx.y;
    int bgd = blockIdx.z;
    if (w >= W || h >= H || bgd >= B * G * D) return;
    int d = bgd % D, g = (bgd / D) % G, b = bgd / (D * G);
    int Cg = C / G, c0 = g * Cg, tw = w - d;
    float sum = 0.0f;
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        sum += __half2float(ref[base + w])
             * ((tw >= 0) ? __half2float(tgt[base + tw]) : 0.0f);
    }
    gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum);
}

template <int DTile>
__device__ __forceinline__ void build_gwc_f16_fusedD_device(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= W || h >= H) return;

    int Cg = C / G;
    int num_dtiles = (D + DTile - 1) / DTile;
    int z = blockIdx.z; // 0 .. B*G*num_dtiles-1
    if (z >= B * G * num_dtiles) return;

    int dt = z % num_dtiles;
    int bg = z / num_dtiles;
    int g = bg % G;
    int b = bg / G;

    int d0 = dt * DTile;
    int c0 = g * Cg;

    float sum[DTile];
#pragma unroll
    for (int i = 0; i < DTile; ++i) sum[i] = 0.0f;

    // For each channel in the group, load ref once and reuse across DTile disparities.
    for (int c = 0; c < Cg; ++c) {
        int base = b*C*H*W + (c0+c)*H*W + h*W;
        float r = __half2float(ref[base + w]);

#pragma unroll
        for (int i = 0; i < DTile; ++i) {
            int d = d0 + i;
            if (d < D) {
                int tw = w - d;
                float t = (tw >= 0) ? __half2float(tgt[base + tw]) : 0.0f;
                sum[i] += r * t;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < DTile; ++i) {
        int d = d0 + i;
        if (d < D) {
            gwc[b*G*D*H*W + g*D*H*W + d*H*W + h*W + w] = __float2half(sum[i]);
        }
    }
}

// Exported wrappers (so other translation units like bench_gwc.cu can link).
extern "C" __global__ void build_gwc_f16_fusedD2(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    build_gwc_f16_fusedD_device<2>(ref, tgt, gwc, B, C, H, W, D, G);
}

extern "C" __global__ void build_gwc_f16_fusedD4(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    build_gwc_f16_fusedD_device<4>(ref, tgt, gwc, B, C, H, W, D, G);
}

extern "C" __global__ void build_gwc_f16_fusedD8(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G)
{
    build_gwc_f16_fusedD_device<8>(ref, tgt, gwc, B, C, H, W, D, G);
}

#include "NvInfer.h"
#include <cuda_runtime.h>

void launchBuildGwc(nvinfer1::DataType type, const void* ref, const void* tgt, void* gwc,
    int B, int C, int H, int W, int D, int G, cudaStream_t stream)
{
    const int BX = 16, BY = 16;
    dim3 block(BX, BY, 1);
    dim3 grid_xy((W + BX - 1) / BX, (H + BY - 1) / BY, 1);

    if (type == nvinfer1::DataType::kHALF) {
        // Fused-disparity kernel: each block computes DTile disparities at once to reuse ref loads.
        // Heuristic: larger tiles reduce ref loads but increase register pressure.
        // D=48 typical; DTile=8 often works well.
        if (D >= 64) {
            int num_dtiles = (D + 8 - 1) / 8;
            dim3 grid(grid_xy.x, grid_xy.y, B * G * num_dtiles);
            build_gwc_f16_fusedD8<<<grid, block, 0, stream>>>(
                static_cast<__half const*>(ref), static_cast<__half const*>(tgt),
                static_cast<__half*>(gwc), B, C, H, W, D, G);
        } else if (D >= 24) {
            int num_dtiles = (D + 8 - 1) / 8;
            dim3 grid(grid_xy.x, grid_xy.y, B * G * num_dtiles);
            build_gwc_f16_fusedD8<<<grid, block, 0, stream>>>(
                static_cast<__half const*>(ref), static_cast<__half const*>(tgt),
                static_cast<__half*>(gwc), B, C, H, W, D, G);
        } else {
            int num_dtiles = (D + 4 - 1) / 4;
            dim3 grid(grid_xy.x, grid_xy.y, B * G * num_dtiles);
            build_gwc_f16_fusedD4<<<grid, block, 0, stream>>>(
                static_cast<__half const*>(ref), static_cast<__half const*>(tgt),
                static_cast<__half*>(gwc), B, C, H, W, D, G);
        }
    } else {
        // Keep float path simple for now.
        dim3 grid(grid_xy.x, grid_xy.y, B * G * D);
        build_gwc_f32<<<grid, block, 0, stream>>>(
            static_cast<float const*>(ref), static_cast<float const*>(tgt), static_cast<__half*>(gwc),
            B, C, H, W, D, G);
    }
}
