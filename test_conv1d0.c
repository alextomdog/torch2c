#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Conv1d: conv1 ================== //
// Weight for conv1d: 2x2x3 @out_channels x in_channels x kernel_size;
float conv1_weights[12] = {-0.39916992, -0.40234375, -0.03323364, -0.28466797, 0.22509766, -0.29467773, 0.35351562, 0.21606445, -0.31738281, -0.30908203, -0.36865234, 0.30615234};
// Bias for conv1d: 2 @out_channels;
float conv1_bias[2] = {-0.25073242, -0.39916992};

// ================== Flatten: fl1 ================== //
// Flatten for layer: fl1;

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @4x10;
float fc1_weight_transposed[40] = {0.32861328, -0.05139160, 0.25903320, 0.42358398, -0.12988281, -0.00244713, -0.14404297, 0.26953125, 0.43237305, 0.16821289, 0.10882568, 0.04724121, 0.27978516, -0.11285400, -0.48779297, -0.37182617, -0.31323242, -0.27075195, -0.30029297, -0.23986816, 0.07373047, 0.08715820, 0.19958496, 0.17065430, 0.40551758, 0.29809570, -0.49169922, -0.08697510, 0.12078857, -0.06896973, 0.33740234, 0.20275879, -0.47241211, -0.17590332, 0.29345703, 0.06616211, 0.26367188, -0.27612305, -0.08947754, -0.33618164};
// Biases for layer: fc1 @10;
float fc1_bias_transposed[10] = {0.23596191, 0.09423828, -0.02784729, -0.47753906, 0.04394531, -0.12060547, -0.49316406, 0.16076660, -0.35888672, -0.46215820};

// ================== Relu: relu ================== //
// Relu for layer: relu;

// ================== Layer: fc2 ================== //
// Transposed weights for layer: fc2 @10x2;
float fc2_weight_transposed[20] = {-0.08563232, -0.21105957, -0.21704102, -0.10498047, -0.26098633, 0.29345703, 0.01252747, -0.17468262, -0.15600586, 0.16772461, 0.01470184, 0.02209473, 0.11407471, 0.12292480, -0.13781738, -0.06506348, -0.13232422, -0.28247070, -0.28881836, -0.14099121};
// Biases for layer: fc2 @2;
float fc2_bias_transposed[2] = {0.11285400, 0.24291992};

// ================== SoftMax: softmax ================== //
// SoftMax for layer: softmax;

// Conv1d 实现
float *Conv1d(int batch_size, int in_channels, int sequence_length, float *input, int out_channels, int kernel_size, int stride, int padding, float *weights, float *bias, bool use_bias)
{
    int output_length = sequence_length - kernel_size + 1;
    float *output_array = (float *)malloc(batch_size * out_channels * output_length * sizeof(float));

    // 手动计算卷积
    for (int n = 0; n < batch_size; ++n)
    { // 批次
        for (int c_out = 0; c_out < out_channels; ++c_out)
        { // 输出通道
            for (int i = 0; i < output_length; ++i)
            { // 输出序列长度
                float conv_sum = 0.0f;
                for (int c_in = 0; c_in < in_channels; ++c_in)
                { // 输入通道
                    int start_idx = c_out * in_channels * kernel_size + c_in * kernel_size;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        int input_idx = n * in_channels * sequence_length + c_in * sequence_length + (i + k);
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

float *forward(float input[], float output[])
{
    float *result_0 = (float *)malloc(sizeof(float) * 16);
    for (int i = 0; i < 16; i++)
    {
        result_0[i] = input[i];
    }
    // conv1_layer
    // conv1_layer(batch_size: 2, in_channels: 2, sequence_length: 4)
    // => @(batch_size: 2, out_channels: 2, out_sequence_length: 2)
    float *result_1 = Conv1d(2, 2, 4, result_0, 2, 3, 1, 0, conv1_weights, conv1_bias, true);
    free(result_0);
    // fl1_layer
    // fc1_layer
    float *result_2 = Linear(2, 4, 10, result_1, fc1_weight_transposed, fc1_bias_transposed);
    free(result_1);
    // relu_relu
    Relu(2, 10, result_2);
    // fc2_layer
    float *result_3 = Linear(2, 10, 2, result_2, fc2_weight_transposed, fc2_bias_transposed);
    free(result_2);
    // softmax_layer
    SoftMax(2, 2, result_3);
    for (int i = 0; i < 4; i++)
    {
        output[i] = result_3[i];
    }
    free(result_3);
}
int main()
{
    float input[16] = {1.2582370042800903, 0.5869812369346619, -0.2915171682834625, -1.4624212980270386, 0.6623755097389221, -0.16948625445365906, -0.7041482925415039, -0.1320120394229889, 0.2812035083770752, -0.025747932493686676, -0.49267578125, -0.40211033821105957, -0.48362234234809875, -0.17247897386550903, -0.6311891674995422, -1.2736942768096924};
    float output[4];
    forward(input, output);
    for (int i = 0; i < 4; i++)
    {
        printf("%f  ", output[i]);
    }
    return 0;
}