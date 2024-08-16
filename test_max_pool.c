#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @5x10;
float fc1_weight_transposed[50] = {0.06707764, 0.21545410, 0.05670166, 0.18737793, 0.26806641, -0.02644348, -0.37402344, -0.15014648, -0.02317810, -0.23034668, -0.15063477, -0.34057617, 0.05737305, 0.13854980, -0.08905029, 0.02157593, 0.11499023, 0.42407227, -0.08850098, -0.30053711, 0.35034180, 0.17224121, -0.12829590, 0.02859497, -0.24365234, -0.21032715, 0.20507812, -0.00494766, -0.36059570, -0.16430664, -0.09960938, 0.14392090, -0.00728607, -0.10546875, -0.35205078, 0.26513672, -0.40600586, -0.25268555, 0.02601624, 0.11822510, 0.23669434, -0.11065674, 0.31542969, 0.44433594, 0.28955078, -0.18823242, -0.10705566, -0.37255859, -0.42187500, 0.06658936};
// Biases for layer: fc1 @10;
float fc1_bias_transposed[10] = {-0.20361328, -0.28784180, 0.43920898, 0.01956177, 0.26806641, -0.14782715, 0.05413818, -0.24523926, 0.18408203, -0.02957153};

// ================== MaxPool1d: max_pool ================== //
// MaxPool1d for layer: max_pool;

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*10);
for (int i = 0; i < 10; i++) { result_0[i] = input[i]; }
// fc1_layer
float* result_1 = Linear(2,5,10,result_0,fc1_weight_transposed,fc1_bias_transposed);
free(result_0);
// max_pool_layer
// output_size: 5
float* result_2 = MaxPool1D(2,10,2,2,0,result_1);
free(result_1);
for (int i = 0; i < 10; i++) { output[i] = result_2[i]; }
	free(result_2);
}
int main(){
float input[10] = { 0.6638058423995972,-0.3219200074672699,0.3757706582546234,-1.0746830701828003,-0.6943925023078918,0.7138792276382446,0.2641783654689789,-1.4406588077545166,-0.6426413059234619,-1.0343194007873535 };
float output[10];
forward(input, output);
for (int i = 0; i < 10; i++){ printf("%f  ", output[i]); 
 }
return 0;
}