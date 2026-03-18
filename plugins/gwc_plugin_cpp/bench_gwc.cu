#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

// Kernel defined in build_gwc_kernel.cu
extern "C" __global__ void build_gwc_f16(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

extern "C" __global__ void build_gwc_f16_fusedD2(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

extern "C" __global__ void build_gwc_f16_fusedD4(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

extern "C" __global__ void build_gwc_f16_fusedD8(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

extern "C" __global__ void build_gwc_f32(
    const float* __restrict__ ref, const float* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

template <typename F>
static float timeKernelMs(F&& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main() {
    // Match typical feature-map dimensions (1/4 of 480x864).
    const int B = 1;
    const int C = 320;
    const int H = 480 / 4;
    const int W = 864 / 4;
    // Fast-FoundationStereo convert.sh uses D=64 (max_disp=256 -> 64 at 1/4).
    const int D = 64;
    const int G = 8;

    const size_t feat_size = static_cast<size_t>(B) * C * H * W;
    const size_t vol_size  = static_cast<size_t>(B) * G * D * H * W;

    std::vector<__half> h_ref(feat_size), h_tgt(feat_size);
    for (size_t i = 0; i < feat_size; ++i) {
        h_ref[i] = __float2half(0.5f);
        h_tgt[i] = __float2half(0.5f);
    }

    __half *d_ref = nullptr, *d_tgt = nullptr, *d_gwc = nullptr;
    cudaMalloc(&d_ref, feat_size * sizeof(__half));
    cudaMalloc(&d_tgt, feat_size * sizeof(__half));
    cudaMalloc(&d_gwc, vol_size  * sizeof(__half));
    cudaMemcpy(d_ref, h_ref.data(), feat_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, h_tgt.data(), feat_size * sizeof(__half), cudaMemcpyHostToDevice);

    // Keep consistent with the plugin launch (see launchBuildGwc).
    const int BLK_X = 16;
    const int BLK_Y = 16;

    dim3 block(BLK_X, BLK_Y, 1);
    dim3 grid((W + BLK_X - 1) / BLK_X,
              (H + BLK_Y - 1) / BLK_Y,
              B * G * D);

    const int warmup = 20;
    const int iters = 200;

    // FP16 baseline (NCHW)
    float f16_base = timeKernelMs([&] {
        build_gwc_f16<<<grid, block, 0>>>(
            d_ref, d_tgt, d_gwc,
            B, C, H, W, D, G);
    }, warmup, iters);

    // FP16 fused-D (NCHW): reuse ref loads across multiple disparities per block
    auto timeFused = [&](int dtile) -> float {
        if (dtile == 2) {
            int num_dtiles = (D + 2 - 1) / 2;
            dim3 grid_fused((W + BLK_X - 1) / BLK_X,
                            (H + BLK_Y - 1) / BLK_Y,
                            B * G * num_dtiles);
            return timeKernelMs([&] {
                build_gwc_f16_fusedD2<<<grid_fused, block, 0>>>(
                    d_ref, d_tgt, d_gwc, B, C, H, W, D, G);
            }, warmup, iters);
        }
        if (dtile == 4) {
            int num_dtiles = (D + 4 - 1) / 4;
            dim3 grid_fused((W + BLK_X - 1) / BLK_X,
                            (H + BLK_Y - 1) / BLK_Y,
                            B * G * num_dtiles);
            return timeKernelMs([&] {
                build_gwc_f16_fusedD4<<<grid_fused, block, 0>>>(
                    d_ref, d_tgt, d_gwc, B, C, H, W, D, G);
            }, warmup, iters);
        }
        // dtile == 8
        int num_dtiles = (D + 8 - 1) / 8;
        dim3 grid_fused((W + BLK_X - 1) / BLK_X,
                        (H + BLK_Y - 1) / BLK_Y,
                        B * G * num_dtiles);
        return timeKernelMs([&] {
            build_gwc_f16_fusedD8<<<grid_fused, block, 0>>>(
                d_ref, d_tgt, d_gwc, B, C, H, W, D, G);
        }, warmup, iters);
    };

    float f16_fusedD2 = timeFused(2);
    float f16_fusedD4 = timeFused(4);
    float f16_fusedD8 = timeFused(8);

    // FP32 (optional; keep identical output type)
    std::vector<float> h_ref_f(feat_size), h_tgt_f(feat_size);
    for (size_t i = 0; i < feat_size; ++i) {
        h_ref_f[i] = 0.5f;
        h_tgt_f[i] = 0.5f;
    }
    float *d_ref_f = nullptr, *d_tgt_f = nullptr;
    cudaMalloc(&d_ref_f, feat_size * sizeof(float));
    cudaMalloc(&d_tgt_f, feat_size * sizeof(float));
    cudaMemcpy(d_ref_f, h_ref_f.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_f, h_tgt_f.data(), feat_size * sizeof(float), cudaMemcpyHostToDevice);

    float f32_base = timeKernelMs([&] {
        build_gwc_f32<<<grid, block, 0>>>(
            d_ref_f, d_tgt_f, d_gwc,
            B, C, H, W, D, G);
    }, warmup, iters);

    printf("Shape: B=%d C=%d H=%d W=%d D=%d G=%d  (grid.z=B*G*D=%d)\n",
           B, C, H, W, D, G, B * G * D);
    printf("Avg over %d iters (ms):\n", iters);
    printf("  FP16 baseline (NCHW)    : %.4f\n", f16_base);
    printf("  FP16 fusedD2 (NCHW)     : %.4f  (vs baseline %.2fx)\n", f16_fusedD2, f16_base / f16_fusedD2);
    printf("  FP16 fusedD4 (NCHW)     : %.4f  (vs baseline %.2fx)\n", f16_fusedD4, f16_base / f16_fusedD4);
    printf("  FP16 fusedD8 (NCHW)     : %.4f  (vs baseline %.2fx)\n", f16_fusedD8, f16_base / f16_fusedD8);

    printf("  FP32 baseline (NCHW)    : %.4f\n", f32_base);
    cudaFree(d_ref_f);
    cudaFree(d_tgt_f);
    cudaFree(d_ref);
    cudaFree(d_tgt);
    cudaFree(d_gwc);

    return 0;
}

