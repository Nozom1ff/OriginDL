#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/normalization/rms_norm.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief RMSNorm 算子测试类（参数化版本）
 */
class RMSNormOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== RMSNorm 前向传播测试 ====================

TEST_P(RMSNormOperatorTest, RMSNorm1DForward)
{
    // 测试 RMSNorm 1D 输入前向传播
    // 输入: (N=2, normalized_shape=3)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor(x_data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 参数: gamma
    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto result = F::rms_norm(x, gamma, 1e-5f);

    // 验证输出形状
    Shape expected_shape{2, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出值
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 6U);

    // 验证输出不为 NaN 或 Inf
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }

    // 手动验证第一行的计算
    // x = [1, 2, 3], RMS = sqrt((1^2 + 2^2 + 3^2) / 3 + eps) = sqrt(14/3 + eps)
    // y[0] = 1 / RMS, y[1] = 2 / RMS, y[2] = 3 / RMS
    float rms = std::sqrt((1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) / 3.0f + 1e-5f);
    EXPECT_NEAR(result_data[0], 1.0f / rms, 1e-4f);
    EXPECT_NEAR(result_data[1], 2.0f / rms, 1e-4f);
    EXPECT_NEAR(result_data[2], 3.0f / rms, 1e-4f);
}

TEST_P(RMSNormOperatorTest, RMSNorm2DForward)
{
    // 测试 RMSNorm 2D 输入前向传播
    // 输入: (N=2, H=2, normalized_shape=2)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto x = Tensor(x_data, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto result = F::rms_norm(x, gamma, 1e-5f);

    // 验证输出形状
    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出不为 NaN 或 Inf
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 8U);

    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormOperatorTest, RMSNorm3DForward)
{
    // 测试 RMSNorm 3D 输入前向传播
    // 输入: (N=2, H=2, W=2, normalized_shape=2)
    std::vector<float> x_data(16, 1.0f);
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        x_data[i] = static_cast<float>(i + 1);
    }
    auto x = Tensor(x_data, Shape{2, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto result = F::rms_norm(x, gamma, 1e-5f);

    // 验证输出形状
    Shape expected_shape{2, 2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出不为 NaN 或 Inf
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 16U);

    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 数值正确性测试 ====================

TEST_P(RMSNormOperatorTest, RMSNormWithGamma)
{
    // 测试 RMSNorm 使用不同 gamma 的情况
    // 输入: (N=1, normalized_shape=3)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    auto x = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // gamma = 2, 所有元素放大 2 倍
    std::vector<float> gamma_data = {2.0f, 2.0f, 2.0f};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::rms_norm(x, gamma, 1e-5f);

    // 验证输出形状
    EXPECT_EQ(result.shape(), Shape({1, 3}));

    // 验证输出值应该是不带 gamma 的 2 倍
    auto result_data = result.to_vector<float>();

    // 计算期望值
    float rms = std::sqrt((1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) / 3.0f + 1e-5f);
    EXPECT_NEAR(result_data[0], 2.0f * 1.0f / rms, 1e-4f);
    EXPECT_NEAR(result_data[1], 2.0f * 2.0f / rms, 1e-4f);
    EXPECT_NEAR(result_data[2], 2.0f * 3.0f / rms, 1e-4f);
}

TEST_P(RMSNormOperatorTest, RMSNormDifferentEps)
{
    // 测试不同 eps 值
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    auto x = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    // 使用较大的 eps
    auto result = F::rms_norm(x, gamma, 1e-3f);

    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 边界情况测试 ====================

TEST_P(RMSNormOperatorTest, RMSNormSingleElement)
{
    // 测试单个元素的情况
    std::vector<float> x_data = {2.0f};
    auto x = Tensor(x_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f};
    auto gamma = Tensor(gamma_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::rms_norm(x, gamma, 1e-5f);

    EXPECT_EQ(result.shape(), Shape({1}));

    // 对于单个元素，输出应该接近 1 (因为 x / sqrt(x^2 + eps) ≈ 1)
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 1.0f, 0.01f);
}

TEST_P(RMSNormOperatorTest, RMSNormZeroInput)
{
    // 测试输入全为零的情况
    std::vector<float> x_data = {0.0f, 0.0f, 0.0f};
    auto x = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::rms_norm(x, gamma, 1e-5f);

    // 当输入全为零时，RMS = sqrt(eps)，输出应该仍为零
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_NEAR(val, 0.0f, 1e-6f);
    }
}

TEST_P(RMSNormOperatorTest, RMSNormLargeValues)
{
    // 测试大值输入
    std::vector<float> x_data = {1000.0f, 2000.0f, 3000.0f};
    auto x = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::rms_norm(x, gamma, 1e-5f);

    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== Float64 类型测试 ====================

TEST_P(RMSNormOperatorTest, RMSNormFloat64)
{
    // 测试 float64 类型
    std::vector<double> x_data = {1.0, 2.0, 3.0};
    auto x = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat64).device(deviceType()));

    std::vector<double> gamma_data = {1.0, 1.0, 1.0};
    auto gamma = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat64).device(deviceType()));

    auto result = F::rms_norm(x, gamma, 1e-5);

    EXPECT_EQ(result.shape(), Shape({1, 3}));

    auto result_data = result.to_vector<double>();
    for (double val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 参数化测试实例化 ====================

INSTANTIATE_DEVICE_TEST_SUITE_P(RMSNormOperatorTest);
