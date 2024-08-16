#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// 对进行归一化
void SoftMax(int batch_size, int elements_length, float *input)
{
    float max_value, sum_exp;

    // 逐行计算softmax
    for (int i = 0; i < batch_size; i++)
    {
        // 找到该行的最大值
        max_value = input[i * elements_length];
        for (int j = 1; j < elements_length; j++)
        {
            if (input[i * elements_length + j] > max_value)
            {
                max_value = input[i * elements_length + j];
            }
        }

        // 计算该行的指数和
        sum_exp = 0.0f;
        for (int j = 0; j < elements_length; j++)
        {
            input[i * elements_length + j] = exp(input[i * elements_length + j] - max_value);
            sum_exp += input[i * elements_length + j];
        }

        // 归一化得到softmax输出
        for (int j = 0; j < elements_length; j++)
        {
            input[i * elements_length + j] /= sum_exp;
        }
    }
}

// 对进行ReLU激活
void Relu(int batch_size, int elements_length, float *input)
{
    for (int i = 0; i < batch_size * elements_length; i++)
    {
        if (input[i] < 0)
        {
            input[i] = 0;
        }
    }
}

float *Linear(int batch_size, int input_size, int output_size, float *input, float *weight_transposed, float *bias)
{
    int i, j, k;

    // 为结果矩阵分配内存
    float *result = (float *)malloc(batch_size * output_size * sizeof(float));
    if (result == NULL)
    {
        printf("Error: Memory allocation failed.\n");
        return NULL;
    }

    // 初始化并执行矩阵乘法
    for (i = 0; i < batch_size; i++)
    {
        for (j = 0; j < output_size; j++)
        {
            result[i * output_size + j] = 0; // 初始化元素为0
            for (k = 0; k < input_size; k++)
            {
                result[i * output_size + j] += input[i * input_size + k] * weight_transposed[k * output_size + j];
            }
            result[i * output_size + j] += bias[j]; // 将偏置项加到结果中
        }
    }

    // 返回结果矩阵
    return result;
}

float *Conv1d(int batch_size, int in_channels, int sequence_length, float *input, int out_channels, int kernel_size, int stride, int padding, float *weights, float *bias, bool use_bias)
{
    // 修正后的 output_length 计算方式
    int output_length = (sequence_length - kernel_size + stride) / stride;

    float *output_array = (float *)malloc(batch_size * out_channels * output_length * sizeof(float));

    // 手动计算卷积
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c_out = 0; c_out < out_channels; ++c_out)
        {
            for (int i = 0; i < output_length; ++i)
            {
                float conv_sum = 0.0f;
                for (int c_in = 0; c_in < in_channels; ++c_in)
                {
                    int start_idx = c_out * in_channels * kernel_size + c_in * kernel_size;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        int input_idx = n * in_channels * sequence_length + c_in * sequence_length + (i * stride + k);
                        int weight_idx = start_idx + k;
                        conv_sum += input[input_idx] * weights[weight_idx];
                    }
                }
                int output_idx = n * out_channels * output_length + c_out * output_length + i;
                output_array[output_idx] = conv_sum;

                // 添加偏置项
                if (use_bias)
                {
                    output_array[output_idx] += bias[c_out];
                }
            }
        }
    }

    return output_array;
}

void BatchNorm1d(
    int batch_size,      // 批量大小
    int elements_length, // 特征的数量（每个样本的特征数量）
    float *input,        // 输入数组，大小为 batch_size * elements_length
    float epsilon,       // 一个很小的数值，防止除零错误
    float *gamma,        // 缩放参数（即PyTorch中的weight），大小为elements_length
    float *beta,         // 平移参数（即PyTorch中的bias），大小为elements_length
    float *running_mean, // 运行中的均值，大小为elements_length
    float *running_var   // 运行中的方差，大小为elements_length
)
{
    // 对每个特征进行标准化
    for (int i = 0; i < elements_length; ++i)
    {
        for (int j = 0; j < batch_size; ++j)
        {
            int idx = j * elements_length + i;
            // 1. 标准化输入
            float normalized = (input[idx] - running_mean[i]) / sqrtf(running_var[i] + epsilon);
            // 2. 应用gamma和beta
            input[idx] = gamma[i] * normalized + beta[i];
        }
    }
}

// MaxPool1D 函数
float *MaxPool1D(int batch_size, int elements_length, int pool_size, int stride, int padding, float *input)
{
    int padded_length = elements_length + 2 * padding;
    int output_length = (padded_length - pool_size) / stride + 1;

    float *output = (float *)malloc(batch_size * output_length * sizeof(float));

    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < output_length; i++)
        {
            int start_index = i * stride - padding;
            float max_value = (start_index >= 0 && start_index < elements_length) ? input[b * elements_length + start_index] : -FLT_MAX;

            for (int j = 1; j < pool_size; j++)
            {
                int index = start_index + j;
                if (index >= 0 && index < elements_length)
                {
                    float value = input[b * elements_length + index];
                    if (value > max_value)
                    {
                        max_value = value;
                    }
                }
            }
            output[b * output_length + i] = max_value;
        }
    }
    return output;
}

