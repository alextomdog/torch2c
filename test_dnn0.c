#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @5x10;
float fc1_weight_transposed[50] = {0.37670898, 0.01902771, -0.31396484, 0.42529297, -0.38452148, 0.34448242, 0.29589844, 0.30053711, -0.32153320, 0.30883789, -0.32421875, 0.15759277, 0.11730957, -0.03050232, -0.23706055, 0.18237305, 0.26611328, 0.28588867, 0.42382812, 0.18444824, -0.32519531, 0.17150879, -0.00329208, 0.00791168, 0.36157227, -0.08178711, -0.40795898, -0.19873047, 0.12365723, 0.07373047, 0.42431641, 0.40112305, -0.13793945, 0.35424805, -0.42504883, -0.05187988, -0.12011719, -0.39331055, -0.34277344, -0.36840820, -0.06250000, 0.30126953, 0.23266602, -0.24365234, -0.02017212, -0.25781250, -0.28076172, 0.00418472, 0.00323486, 0.03988647};
// Biases for layer: fc1 @10;
float fc1_bias_transposed[10] = {-0.36474609, 0.21032715, 0.16394043, -0.38940430, 0.15051270, 0.17077637, -0.05245972, 0.26684570, -0.34155273, 0.15893555};

// ================== Relu: relu1 ================== //
// Relu for layer: relu1;

// ================== Layer: fc2 ================== //
// Transposed weights for layer: fc2 @10x20;
float fc2_weight_transposed[200] = {-0.20202637, -0.11364746, -0.23059082, -0.02740479, 0.12622070, -0.28735352, -0.15063477, 0.18811035, 0.11157227, 0.02915955, -0.16271973, 0.08416748, 0.22778320, -0.04910278, -0.05184937, 0.03054810, -0.30004883, 0.12213135, 0.29150391, 0.13085938, 0.29296875, -0.00013781, 0.11529541, -0.28271484, -0.21142578, 0.16918945, 0.25024414, -0.28198242, -0.16699219, -0.15490723, 0.16174316, -0.09582520, -0.01734924, 0.12976074, 0.11926270, -0.28149414, 0.22045898, 0.27148438, -0.20446777, 0.14831543, 0.01440430, -0.15771484, -0.12103271, 0.19226074, -0.29858398, -0.21643066, 0.06384277, -0.09228516, 0.30297852, -0.18566895, 0.19738770, 0.21545410, -0.22631836, 0.01369476, -0.17419434, 0.05364990, 0.26391602, -0.06817627, -0.12707520, 0.29199219, -0.17199707, -0.28442383, -0.10394287, -0.22375488, 0.12890625, 0.12951660, 0.28930664, 0.07604980, -0.23754883, -0.24877930, -0.26171875, -0.10717773, -0.00787354, -0.01443481, -0.04278564, -0.17443848, 0.15930176, 0.17541504, 0.14636230, 0.24829102, -0.30932617, 0.16125488, 0.03155518, -0.29858398, 0.30688477, -0.15930176, 0.14611816, -0.04702759, -0.25463867, -0.20886230, 0.13574219, 0.04705811, 0.17773438, 0.25195312, -0.09234619, 0.05166626, 0.26928711, -0.05712891, -0.00493622, -0.23168945, 0.13452148, 0.28784180, -0.00994873, 0.30981445, -0.16467285, 0.22631836, 0.25463867, -0.08935547, -0.06512451, 0.25708008, 0.10888672, 0.11602783, -0.07031250, 0.09667969, -0.28491211, 0.20422363, 0.19836426, -0.16125488, -0.07934570, -0.08886719, -0.03906250, -0.20141602, 0.30297852, 0.15966797, -0.30151367, -0.15319824, 0.02134705, 0.11163330, 0.03283691, -0.02456665, -0.15649414, -0.07324219, -0.25659180, 0.26464844, -0.03247070, -0.07202148, 0.24829102, -0.11981201, 0.28979492, 0.11883545, -0.08984375, 0.01800537, 0.00073195, -0.03961182, -0.25244141, 0.24548340, 0.26293945, 0.01943970, -0.26635742, 0.11260986, -0.14990234, 0.06634521, -0.06262207, 0.27514648, -0.26440430, -0.08020020, 0.03536987, 0.27197266, 0.05886841, -0.23767090, 0.00550842, 0.09460449, 0.18469238, -0.29077148, -0.00361824, -0.09442139, -0.17639160, -0.00402832, -0.15222168, 0.00286293, -0.06652832, 0.24084473, 0.24829102, -0.02886963, 0.03460693, 0.17663574, -0.20227051, 0.01918030, 0.03045654, -0.03366089, 0.03591919, -0.12951660, 0.05050659, 0.18469238, -0.24719238, 0.30004883, -0.25634766, -0.07318115, -0.18054199, 0.08251953, 0.14404297, -0.27441406, -0.28271484, -0.28491211, 0.24047852, -0.27539062, -0.14709473, 0.01297760, -0.16821289, 0.24291992};
// Biases for layer: fc2 @20;
float fc2_bias_transposed[20] = {-0.16918945, -0.25024414, 0.24719238, -0.24548340, -0.19372559, 0.04916382, -0.09960938, -0.10455322, 0.05130005, 0.13562012, -0.03001404, -0.19604492, -0.03945923, 0.09271240, -0.30737305, -0.26928711, -0.01058197, 0.15905762, -0.20275879, -0.21704102};

// ================== SoftMax: softmax ================== //
// SoftMax for layer: softmax;

// ================== Layer: fc5 ================== //
// Transposed weights for layer: fc5 @20x3;
float fc5_weight_transposed[60] = {-0.02661133, -0.07849121, 0.16638184, -0.16113281, 0.13317871, 0.01004791, -0.07696533, 0.15466309, 0.03274536, -0.15087891, -0.07415771, -0.09942627, 0.07006836, -0.00308800, 0.14916992, 0.14245605, 0.18347168, 0.07916260, 0.19409180, 0.21215820, 0.05575562, -0.04888916, -0.21801758, -0.13879395, 0.20959473, 0.04217529, -0.21459961, -0.21533203, 0.21020508, -0.12457275, 0.00736237, 0.01576233, 0.05847168, -0.12866211, 0.16979980, -0.10852051, -0.12036133, -0.07696533, -0.17077637, 0.16467285, -0.20617676, -0.02836609, 0.11993408, -0.20996094, -0.12475586, -0.05920410, -0.21398926, 0.01232147, -0.22070312, 0.08428955, 0.03387451, -0.05810547, -0.20959473, 0.09356689, -0.09307861, 0.19311523, 0.18029785, 0.07434082, 0.11273193, 0.15112305};
// Biases for layer: fc5 @3;
float fc5_bias_transposed[3] = {0.07824707, 0.00751495, 0.07427979};

// ================== SoftMax: softmax2 ================== //
// SoftMax for layer: softmax2;

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
    float *result_0 = (float *)malloc(sizeof(float) * 10);
    for (int i = 0; i < 10; i++)
    {
        result_0[i] = input[i];
    }
    // fc1_layer
    float *result_1 = Linear(2, 5, 10, result_0, fc1_weight_transposed, fc1_bias_transposed);
    free(result_0);
    // relu1_relu
    Relu(2, 10, result_1);
    // fc2_layer
    float *result_2 = Linear(2, 10, 20, result_1, fc2_weight_transposed, fc2_bias_transposed);
    free(result_1);
    // softmax_layer
    SoftMax(2, 20, result_2);
    // fc5_layer
    float *result_3 = Linear(2, 20, 3, result_2, fc5_weight_transposed, fc5_bias_transposed);
    free(result_2);
    // softmax2_layer
    SoftMax(2, 3, result_3);
    for (int i = 0; i < 6; i++)
    {
        output[i] = result_3[i];
    }
    free(result_3);
}
int main()
{
    float input[10] = {1.3020135164260864, -0.1191926822066307, -0.8468743562698364, 0.9985636472702026, 0.1667756736278534, 0.22539983689785004, -0.5317959189414978, -1.0635405778884888, -1.0320861339569092, -0.33052170276641846};
    float output[6];
    forward(input, output);
    for (int i = 0; i < 6; i++)
    {
        printf("%f  ", output[i]);
    }
    return 0;
}