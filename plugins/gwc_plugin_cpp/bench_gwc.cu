#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

// Kernel defined in build_gwc_kernel.cu
extern "C" __global__ void build_gwc_f16(
    const __half* __restrict__ ref, const __half* __restrict__ tgt,
    __half* __restrict__ gwc,
    int B, int C, int H, int W, int D, int G);

int main() {
    // Match typical feature-map dimensions (1/4 of 480x864, max_disp=192 -> D=48)
    const int B = 1;
    const int C = 320;
    const int H = 480 / 4;
    const int W = 864 / 4;
    const int D = 48;
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

    const int BLK_X = 32;
    const int BLK_Y = 8;

    dim3 block(BLK_X, BLK_Y, 1);
    dim3 grid((W + BLK_X - 1) / BLK_X,
              (H + BLK_Y - 1) / BLK_Y,
              B * G);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        build_gwc_f16<<<grid, block, BLK_Y * (BLK_X + D - 1) * sizeof(float)>>>(
            d_ref, d_tgt, d_gwc,
            B, C, H, W, D, G);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int iters = 100;

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        build_gwc_f16<<<grid, block, BLK_Y * (BLK_X + D - 1) * sizeof(float)>>>(
            d_ref, d_tgt, d_gwc,
            B, C, H, W, D, G);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Avg build_gwc_f16 kernel time over %d iters: %.3f ms\n", iters, ms / iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_ref);
    cudaFree(d_tgt);
    cudaFree(d_gwc);

    return 0;
}

