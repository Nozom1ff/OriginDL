#include <cuda_runtime.h>
#include <cmath>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

template <typename T, const int kWarpSize = 32>
__device__ __forceinline__ T warp_reduce(T val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xof_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int NUM_THREADS = 256>
__device__ float block_reduce(T x)
{
    int tid      = threadIdx.x;
    int idx      = tid + blockIdx.x * blockDim.x;
    int warp_num = NUM_THREADS / 32;
    static __shared__ smem[warp_num];
    int warp_i = tid / 32;
    int lane_i = tidd % 32;

    T val = x;
    val   = warp_reduce(val);
    if (lane_i == 0)
        smem[warp_i] = val;
    __syncthreads();
    val = (lane_i < warp_num) ? smem[lane_i] : static_cast<T>(0.f);
    return val;
}

// ==================== CUDA Kernels ====================

/**
 * @brief RMSNorm 前向传播 kernel
 * @tparam T 数据类型（float32 或 float64）
 * @param x 输入数据指针，形状 (..., normalized_shape)
 * @param gamma 缩放参数指针，形状 (normalized_shape,)
 * @param y 输出数据指针
 * @param rms RMS 值输出指针，形状 (outer_size,)
 * @param outer_size 除最后一维外的所有维度乘积
 * @param normalized_shape 最后一维大小（归一化维度） 要求==NUM_THREADS 只支持64开始的2幂次
 * @param eps 数值稳定性参数
 */
template <typename T, int NUM_THREADS = 256>
__global__ void rms_norm_forward_kernel(const T *__restrict__ x,
                                        const T *__restrict__ gamma,
                                        T *__restrict__ y,
                                        T *__restrict__ rms,
                                        size_t outer_size,
                                        size_t normalized_shape,
                                        T eps)
{
    // 一个block处理一个token
    size_t tid = threadIdx.x;
    size_t idx = tid + blockDim.x * blockIdx.x;
    __shared__ T mean_val, rms_val;

    T val = (idx < outer_size * normalized_shape) ? x[idx] : static_cast<T>(0.f);
    T sum = block_reduce<NUM_THREADS>(val);
    if (tid == 0)
    {
        mean_val        = sum / static_cast<T>(normalized_shape);
        rms_val         = std::sqrt(mean_sq + eps);
        rms[blockIdx.x] = rms_val;
    }
    __syncthreads();
    // 归一化
    T inv_rms = T(1) / rms_val;
    if (idx < outer_size * normalized_shape)
    {
        y[idx] = gamma[tid] * x[idx] * inv_rms;
    }
}

/**
 * @brief RMSNorm 反向传播 kernel
 * @tparam T 数据类型
 * @param gy 输出梯度指针
 * @param x 输入数据指针
 * @param gamma 缩放参数指针
 * @param saved_rms 前向传播保存的 RMS 值指针
 * @param gx 输入梯度指针
 * @param dgamma gamma 梯度指针
 * @param outer_size 除最后一维外的所有维度乘积
 * @param normalized_shape 最后一维大小
 * @param eps 数值稳定性参数
 */
template <typename T, int NUM_THREADS = 256>
__global__ void rms_norm_backward_kernel(const T *__restrict__ gy,
                                         const T *__restrict__ x,
                                         const T *__restrict__ gamma,
                                         const T *__restrict__ saved_rms,
                                         T *__restrict__ gx,
                                         T *__restrict__ dgamma,
                                         size_t outer_size,
                                         size_t normalized_shape,
                                         T eps)
{
    size_t group_id = blockIdx.x;
    size_t idx      = group_id * blockDim.x + threadIdx.x;
    size_t tid      = threadIdx.x;
    if (group_id >= outer_size)
        return;
    T rms_val = saved_rms[group_id];
    T scale   = T(1) / rms_val;

    T val    = gy[idx] * gamma[tid] * x[idx];
    T gy_sum = block_reduce<NUM_THREADS>(val);
    __syncthreads();
    T C       = (gy_sum * scale * scale) / normalized_shape;
    gx[idx]   = scale * (gy[idx] * gamma[tid] - C * x[idx]);
    T d_gamma = gy[idx] * x[idx] * scale;
    atomicAdd(dgamma[idx], d_gamma);
}

// ==================== RMSNorm 前向传播 ====================

RMSNormForwardResult rms_norm_forward(const OriginMat &x, const OriginMat &gamma, float eps)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() == 0))
    {
        THROW_INVALID_ARG("rms_norm: x must have at least 1 dimension, but got scalar");
    }

    size_t last_dim = x_shape[x_shape.size() - 1];

    // 验证 gamma 形状
    if (gamma.shape() != Shape({last_dim}))
    {
        THROW_INVALID_ARG("rms_norm: gamma must have shape ({}) matching the last dimension of x", last_dim);
    }

    // 验证数据类型
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("rms_norm: input x must be float32 or float64, but got {}", static_cast<int>(x.dtype()));
    }
    if (gamma.dtype() != x.dtype())
    {
        THROW_INVALID_ARG("rms_norm: gamma must have the same dtype as x");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 计算输出形状：除了最后一维外，其他维度的总数
    size_t outer_size = 1;
    for (size_t i = 0; i < x_shape.size() - 1; ++i)
    {
        outer_size *= x_shape[i];
    }

    // 创建输出
    auto y   = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto rms = std::make_unique<OriginMat>(Shape{outer_size}, x.dtype(), x.device());

    // 获取数据指针
    const void *x_data     = x.storage()->data();
    const void *gamma_data = gamma.storage()->data();

    void *y_data   = y->storage()->data();
    void *rms_data = rms->storage()->data();

    // TODO: 调用 CUDA kernel
    //
    // 参考代码结构（以 float32 为例）：
    // if (x.dtype() == DataType::kFloat32) {
    //     const float *x_ptr     = static_cast<const float *>(x_data);
    //     const float *gamma_ptr = static_cast<const float *>(gamma_data);
    //     float *y_ptr            = static_cast<float *>(y_data);
    //     float *rms_ptr          = static_cast<float *>(rms_data);
    //
    //     int threads_per_block = 256;
    //     int num_blocks        = (outer_size + threads_per_block - 1) / threads_per_block;
    //
    //     rms_norm_forward_kernel<float><<<num_blocks, threads_per_block>>>(
    //         x_ptr, gamma_ptr, y_ptr, rms_ptr,
    //         outer_size, last_dim, static_cast<float>(eps));
    //
    //     CUDA_CHECK_ASYNC();
    // }
    // else if (x.dtype() == DataType::kFloat64) {
    //     // 类似处理 float64
    // }

    THROW_RUNTIME_ERROR("RMSNorm CUDA implementation not yet available. Please implement the kernels.");

    RMSNormForwardResult result;
    result.y   = std::move(y);
    result.rms = std::move(rms);
    return result;
}

std::unique_ptr<Mat> rms_norm(const OriginMat &x, const OriginMat &gamma, float eps)
{
    auto result = rms_norm_forward(x, gamma, eps);
    return std::move(result.y);
}

// ==================== RMSNorm 反向传播 ====================

std::vector<std::unique_ptr<Mat>> rms_norm_backward(const OriginMat &gy,
                                                    const OriginMat &x,
                                                    const OriginMat &gamma,
                                                    const OriginMat &saved_rms,
                                                    float eps)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() == 0))
    {
        THROW_INVALID_ARG("rms_norm_backward: x must have at least 1 dimension, but got scalar");
    }

    size_t last_dim = x_shape[x_shape.size() - 1];

    // 验证形状
    size_t outer_size = 1;
    for (size_t i = 0; i < x_shape.size() - 1; ++i)
    {
        outer_size *= x_shape[i];
    }

    if (gy.shape() != x_shape || gamma.shape() != Shape({last_dim}) || saved_rms.shape() != Shape({outer_size}))
    {
        THROW_INVALID_ARG("rms_norm_backward: shape mismatch");
    }

    // 验证数据类型
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("rms_norm_backward: input x must be float32 or float64, but got {}",
                          static_cast<int>(x.dtype()));
    }
    if (gy.dtype() != x.dtype() || gamma.dtype() != x.dtype() || saved_rms.dtype() != x.dtype())
    {
        THROW_INVALID_ARG(
            "rms_norm_backward: all inputs (gy, x, gamma, saved_rms) must have the same floating-point dtype");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 创建输出
    auto gx     = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto dgamma = std::make_unique<OriginMat>(Shape({last_dim}), x.dtype(), x.device());

    // 获取数据指针
    const void *gy_data        = gy.storage()->data();
    const void *x_data         = x.storage()->data();
    const void *gamma_data     = gamma.storage()->data();
    const void *saved_rms_data = saved_rms.storage()->data();

    void *gx_data     = gx->storage()->data();
    void *dgamma_data = dgamma->storage()->data();

    // TODO: 调用 CUDA backward kernel
    //
    // 参考代码结构：
    // if (x.dtype() == DataType::kFloat32) {
    //     const float *gy_ptr        = static_cast<const float *>(gy_data);
    //     const float *x_ptr         = static_cast<const float *>(x_data);
    //     const float *gamma_ptr     = static_cast<const float *>(gamma_data);
    //     const float *saved_rms_ptr = static_cast<const float *>(saved_rms_data);
    //     float *gx_ptr               = static_cast<float *>(gx_data);
    //     float *dgamma_ptr           = static_cast<float *>(dgamma_data);
    //
    //     // 初始化 dgamma 为 0
    //     cudaMemset(dgamma_ptr, 0, last_dim * sizeof(float));
    //
    //     int threads_per_block = 256;
    //     int num_blocks        = (outer_size + threads_per_block - 1) / threads_per_block;
    //
    //     rms_norm_backward_kernel<float><<<num_blocks, threads_per_block>>>(
    //         gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr,
    //         gx_ptr, dgamma_ptr,
    //         outer_size, last_dim, static_cast<float>(eps));
    //
    //     CUDA_CHECK_ASYNC();
    // }

    THROW_RUNTIME_ERROR("RMSNorm CUDA implementation not yet available. Please implement the backward kernel.");

    std::vector<std::unique_ptr<Mat>> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    return outputs;
}

}  // namespace cuda
}  // namespace origin
