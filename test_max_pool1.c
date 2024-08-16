#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @5x10;
float fc1_weight_transposed[50] = {-0.11785889, 0.05197144, -0.20324707, 0.22497559, -0.27294922, -0.38427734, 0.01148224, -0.13452148, 0.35229492, 0.39843750, 0.28076172, 0.08068848, -0.30541992, 0.11529541, 0.16418457, -0.13171387, -0.37475586, 0.03768921, 0.32812500, 0.26928711, 0.31372070, 0.12683105, -0.31152344, 0.32983398, -0.02476501, -0.19213867, -0.34399414, 0.15417480, -0.10595703, 0.27856445, 0.41845703, 0.07250977, -0.42309570, 0.37011719, 0.22851562, -0.39184570, -0.34106445, 0.38232422, 0.16210938, -0.32495117, -0.37890625, 0.26318359, 0.30273438, -0.44018555, -0.03686523, 0.38476562, 0.29199219, 0.02838135, 0.25048828, 0.15502930};
// Biases for layer: fc1 @10;
float fc1_bias_transposed[10] = {-0.11639404, -0.06719971, -0.25366211, -0.11614990, 0.30102539, -0.11950684, -0.35009766, 0.35156250, -0.17126465, 0.25292969};

// ================== BatchNorm1d: batch_normal1d ================== //
// BatchNorm1d for layer: batch_normal1d;
// epsilon for BatchNorm1d: 1e-05;
float batch_normal1d_batch_normal1d_epsilon = 1e-05;
// gamma for BatchNorm1d: 10;
float batch_normal1d_batch_normal1d_gamma[10] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};
// beta for BatchNorm1d: 10;
float batch_normal1d_batch_normal1d_beta[10] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_mean for BatchNorm1d: 10;
float batch_normal1d_batch_normal1d_running_mean[10] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_var for BatchNorm1d: 10;
float batch_normal1d_batch_normal1d_running_var[10] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};

// ================== MaxPool1d: max_pool1 ================== //
// MaxPool1d for layer: max_pool1;

// ================== Relu: relu ================== //
// Relu for layer: relu;

// ================== Layer: fc2 ================== //
// Transposed weights for layer: fc2 @5x3;
float fc2_weight_transposed[15] = {0.08666992, 0.07958984, 0.25903320, 0.19921875, -0.09417725, -0.44458008, -0.30395508, 0.04473877, 0.32934570, 0.13574219, 0.33984375, 0.22802734, 0.36425781, 0.28125000, -0.35253906};
// Biases for layer: fc2 @3;
float fc2_bias_transposed[3] = {0.31494141, -0.04879761, -0.38745117};

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*10);
for (int i = 0; i < 10; i++) { result_0[i] = input[i]; }
// fc1_layer
float* result_1 = Linear(2,5,10,result_0,fc1_weight_transposed,fc1_bias_transposed);
free(result_0);
// batch_normal1d_layer
BatchNorm1d(2,10,result_1,batch_normal1d_batch_normal1d_epsilon,batch_normal1d_batch_normal1d_gamma,batch_normal1d_batch_normal1d_beta, batch_normal1d_batch_normal1d_running_mean,batch_normal1d_batch_normal1d_running_var);
// max_pool1_layer
// output_size: 5
float* result_2 = MaxPool1D(2,10,2,2,0,result_1);
free(result_1);
// relu_relu
Relu(2,5,result_2);
// fc2_layer
float* result_3 = Linear(2,5,3,result_2,fc2_weight_transposed,fc2_bias_transposed);
free(result_2);
for (int i = 0; i < 6; i++) { output[i] = result_3[i]; }
	free(result_3);
}
int main(){
float input[10] = { -0.6244495511054993,0.937702476978302,-1.8548610210418701,0.29335159063339233,-2.749600887298584,-1.0127259492874146,0.7881359457969666,-1.1204769611358643,0.040858082473278046,0.20865951478481293 };
float output[6];
forward(input, output);
for (int i = 0; i < 6; i++){ printf("%f  ", output[i]); 
 }
return 0;
}