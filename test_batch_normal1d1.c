#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @5x10;
float fc1_weight_transposed[50] = {-0.41113281, -0.28808594, -0.17834473, -0.43115234, 0.16040039, 0.29101562, -0.38549805, 0.10217285, -0.26977539, -0.17590332, 0.09985352, -0.22241211, -0.14489746, -0.01649475, -0.40185547, -0.30688477, 0.23095703, 0.22497559, 0.19714355, 0.23425293, 0.08203125, 0.15539551, 0.21350098, 0.22131348, 0.23156738, -0.27441406, -0.43750000, -0.14916992, 0.30346680, -0.15759277, 0.34985352, -0.16577148, -0.37890625, 0.32934570, -0.04650879, -0.24511719, 0.21374512, -0.04425049, -0.19543457, -0.27905273, 0.40136719, -0.24719238, 0.43994141, 0.28344727, 0.04171753, 0.32714844, -0.28051758, -0.26489258, -0.41015625, 0.29882812};
// Biases for layer: fc1 @10;
float fc1_bias_transposed[10] = {-0.16308594, -0.04003906, -0.00040102, -0.25341797, 0.13684082, -0.15100098, -0.06579590, -0.44555664, -0.29785156, 0.43920898};

// ================== BatchNorm1d: batch_normal_1d ================== //
// BatchNorm1d for layer: batch_normal_1d;
// epsilon for BatchNorm1d: 1e-05;
float batch_normal_1d_batch_normal1d_epsilon = 1e-05;
// gamma for BatchNorm1d: 10;
float batch_normal_1d_batch_normal1d_gamma[10] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};
// beta for BatchNorm1d: 10;
float batch_normal_1d_batch_normal1d_beta[10] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_mean for BatchNorm1d: 10;
float batch_normal_1d_batch_normal1d_running_mean[10] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_var for BatchNorm1d: 10;
float batch_normal_1d_batch_normal1d_running_var[10] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};

// ================== Dropout: dropout1 ================== //
// Dropout for layer: dropout1;

// ================== Layer: fc2 ================== //
// Transposed weights for layer: fc2 @10x20;
float fc2_weight_transposed[200] = {0.06198120, -0.21936035, -0.10711670, 0.18908691, 0.25537109, -0.00407028, -0.28735352, 0.15771484, 0.13562012, 0.07415771, -0.00822449, -0.09631348, -0.16271973, -0.21594238, -0.09259033, 0.25292969, 0.23425293, -0.30444336, 0.02522278, -0.26318359, -0.15148926, 0.19641113, -0.15466309, 0.08630371, 0.29272461, 0.26977539, -0.28295898, -0.30224609, 0.13085938, -0.01080322, -0.13696289, 0.04824829, -0.00325775, -0.15881348, 0.26806641, 0.02635193, 0.23034668, 0.30590820, 0.11828613, 0.01829529, -0.03240967, -0.23852539, -0.03152466, 0.09515381, -0.23901367, -0.06756592, -0.30297852, -0.19628906, -0.02441406, 0.10076904, -0.04440308, -0.19470215, 0.04870605, -0.15270996, 0.03952026, 0.08074951, 0.24572754, -0.11761475, -0.27465820, 0.06045532, -0.02369690, 0.24499512, 0.15283203, 0.17883301, 0.22888184, -0.04629517, 0.27807617, 0.15954590, 0.15063477, 0.08453369, 0.13500977, -0.26440430, -0.22949219, -0.27343750, -0.09448242, 0.09887695, 0.07250977, 0.27148438, 0.03677368, 0.12255859, 0.20153809, -0.02587891, -0.08660889, -0.19030762, 0.01150513, 0.09350586, -0.27441406, 0.02664185, 0.17590332, 0.00455856, 0.07513428, 0.14978027, 0.15551758, 0.26977539, 0.12347412, -0.28369141, 0.16003418, -0.08190918, -0.13659668, 0.26928711, -0.22363281, 0.26391602, 0.06222534, 0.02090454, 0.26074219, 0.11730957, 0.18579102, -0.26660156, 0.25903320, -0.13439941, -0.20959473, 0.29370117, -0.13940430, -0.05682373, -0.10406494, 0.28442383, 0.27148438, -0.15307617, 0.01294708, 0.24682617, 0.12066650, -0.29272461, 0.01216125, 0.17150879, -0.16931152, 0.27465820, 0.15563965, -0.10205078, -0.20007324, -0.24597168, 0.26489258, -0.03436279, 0.14562988, 0.10852051, 0.08276367, 0.19287109, -0.03875732, 0.10577393, -0.05105591, 0.30664062, 0.19116211, 0.07348633, 0.05413818, 0.07733154, 0.11462402, 0.08819580, 0.21203613, 0.01246643, -0.30786133, 0.17663574, 0.14123535, -0.23645020, 0.27758789, -0.11975098, 0.27026367, -0.30444336, 0.19909668, 0.10064697, -0.15759277, 0.10357666, 0.31250000, -0.08978271, 0.31591797, 0.09625244, 0.12432861, -0.17272949, 0.28051758, 0.08044434, -0.24877930, -0.29467773, -0.14587402, -0.11627197, 0.13488770, -0.03405762, 0.16271973, 0.09295654, -0.06579590, 0.04824829, 0.13085938, -0.13012695, -0.28735352, 0.01004791, -0.06066895, -0.16516113, -0.23815918, -0.20727539, 0.13354492, -0.12219238, 0.29541016, -0.23706055, 0.07000732, -0.05206299, -0.05538940, 0.09558105, -0.16320801, 0.29199219, -0.20849609, -0.03274536, -0.09185791, -0.27465820};
// Biases for layer: fc2 @20;
float fc2_bias_transposed[20] = {-0.13171387, -0.21350098, -0.01823425, 0.13940430, 0.20104980, 0.09942627, -0.19726562, -0.23754883, 0.17211914, -0.09100342, -0.10577393, 0.11285400, 0.22741699, 0.12561035, 0.26586914, 0.05944824, 0.01454926, 0.08673096, -0.08801270, -0.03527832};

// ================== Relu: relu ================== //
// Relu for layer: relu;

// ================== Layer: fc5 ================== //
// Transposed weights for layer: fc5 @20x3;
float fc5_weight_transposed[60] = {-0.13757324, -0.11743164, 0.01451111, 0.01661682, 0.14013672, 0.14062500, 0.06030273, 0.09375000, -0.17541504, 0.08843994, 0.07629395, -0.21838379, -0.08374023, 0.06274414, -0.06213379, 0.09411621, 0.08013916, -0.04605103, -0.21545410, -0.05398560, -0.21032715, 0.00234413, 0.02218628, 0.12524414, 0.04635620, 0.20678711, -0.04373169, 0.17248535, -0.18530273, -0.00584030, -0.15820312, -0.20532227, 0.15612793, 0.16955566, 0.18847656, 0.13500977, 0.06787109, -0.03540039, -0.18920898, 0.10382080, 0.05206299, 0.09411621, -0.04995728, 0.12695312, 0.20019531, 0.07739258, -0.08453369, -0.03475952, 0.20751953, -0.18151855, -0.11267090, 0.01866150, 0.14721680, -0.09497070, 0.19860840, -0.01403046, 0.09100342, -0.13781738, 0.12939453, -0.18823242};
// Biases for layer: fc5 @3;
float fc5_bias_transposed[3] = {0.21130371, 0.17163086, -0.15344238};

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*5);
for (int i = 0; i < 5; i++) { result_0[i] = input[i]; }
// fc1_layer
float* result_1 = Linear(1,5,10,result_0,fc1_weight_transposed,fc1_bias_transposed);
free(result_0);
// batch_normal_1d_layer
BatchNorm1d(1,10,result_1,batch_normal_1d_batch_normal1d_epsilon,batch_normal_1d_batch_normal1d_gamma,batch_normal_1d_batch_normal1d_beta, batch_normal_1d_batch_normal1d_running_mean,batch_normal_1d_batch_normal1d_running_var);
// dropout1_layer
// fc2_layer
float* result_2 = Linear(1,10,20,result_1,fc2_weight_transposed,fc2_bias_transposed);
free(result_1);
// relu_relu
Relu(1,20,result_2);
// fc5_layer
float* result_3 = Linear(1,20,3,result_2,fc5_weight_transposed,fc5_bias_transposed);
free(result_2);
// softmax2_layer
SoftMax(1,3,result_3);
for (int i = 0; i < 3; i++) { output[i] = result_3[i]; }
	free(result_3);
}
int main(){
float input[5] = { -1.6667474508285522,-0.20194470882415771,1.516796350479126,-1.4839059114456177,-0.7363266944885254 };
float output[3];
forward(input, output);
for (int i = 0; i < 3; i++){ printf("%f  ", output[i]); 
 }
return 0;
}