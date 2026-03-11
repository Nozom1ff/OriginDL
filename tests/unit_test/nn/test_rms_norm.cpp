#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/nn/layers/rms_norm.h"
#include "test_utils.h"

using namespace origin;
namespace nn = origin::nn;

class RMSNormLayerTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(RMSNormLayerTest, BasicForward1D)
{
    // 测试基本的 RMSNorm 层前向传播 - 1D 输入
    nn::RMSNorm rms_norm(4, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, normalized_shape=4)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto x = Tensor(x_data, Shape{2, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, BasicForward2D)
{
    // 测试 RMSNorm 层前向传播 - 2D 输入 (N, H, normalized_shape)
    nn::RMSNorm rms_norm(3, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, H=4, normalized_shape=3)
    std::vector<float> x_data(2 * 4 * 3, 1.0f);
    auto x = Tensor(x_data, Shape{2, 4, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 3U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);
    EXPECT_EQ(y.shape()[2], 3U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, BasicForward3D)
{
    // 测试 RMSNorm 层前向传播 - 3D 输入 (N, H, W, normalized_shape)
    nn::RMSNorm rms_norm(2, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, H=3, W=4, normalized_shape=2)
    std::vector<float> x_data(2 * 3 * 4 * 2, 1.0f);
    auto x = Tensor(x_data, Shape{2, 3, 4, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 3U);
    EXPECT_EQ(y.shape()[2], 4U);
    EXPECT_EQ(y.shape()[3], 2U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, SingleElement)
{
    // 测试单个元素的情况
    nn::RMSNorm rms_norm(1, 1e-5f);
    rms_norm.to(Device(deviceType()));

    std::vector<float> x_data = {2.0f};
    auto x = Tensor(x_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({1}));

    // 对于单个元素，RMSNorm 应该输出 gamma * x / sqrt(x^2 + eps)
    // gamma 初始化为 1，所以输出应该接近 1 (因为 x / sqrt(x^2 + eps) ≈ 1)
    auto y_data = y.to_vector<float>();
    EXPECT_NEAR(y_data[0], 1.0f, 0.01f);
}

TEST_P(RMSNormLayerTest, ResetParameters)
{
    // 测试参数重置
    nn::RMSNorm rms_norm(4, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 修改 gamma 参数
    auto weight = rms_norm.weight();
    auto weight_data = weight->data_ptr<float>();
    weight_data[0] = 2.0f;

    // 重置参数
    rms_norm.reset_parameters();

    // 验证 gamma 被重置为全 1
    auto reset_weight = rms_norm.weight();
    auto reset_data = reset_weight->to_vector<float>();
    for (float val : reset_data)
    {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(RMSNormLayerTests, RMSNormLayerTest, ::testing::Values(DeviceType::kCPU));
