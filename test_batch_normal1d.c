#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== BatchNorm1d: batch_normal_1d ================== //
// BatchNorm1d for layer: batch_normal_1d;
// epsilon for BatchNorm1d: 1e-05;
float batch_normal_1d_batch_normal1d_epsilon = 1e-05;
// gamma for BatchNorm1d: 5;
float batch_normal_1d_batch_normal1d_gamma[5] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};
// beta for BatchNorm1d: 5;
float batch_normal_1d_batch_normal1d_beta[5] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_mean for BatchNorm1d: 5;
float batch_normal_1d_batch_normal1d_running_mean[5] = {0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000};
// running_var for BatchNorm1d: 5;
float batch_normal_1d_batch_normal1d_running_var[5] = {1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000};

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*5);
for (int i = 0; i < 5; i++) { result_0[i] = input[i]; }
// batch_normal_1d_layer
BatchNorm1d(1,5,result_0,batch_normal_1d_batch_normal1d_epsilon,batch_normal_1d_batch_normal1d_gamma,batch_normal_1d_batch_normal1d_beta, batch_normal_1d_batch_normal1d_running_mean,batch_normal_1d_batch_normal1d_running_var);
for (int i = 0; i < 5; i++) { output[i] = result_0[i]; }
	free(result_0);
}
int main(){
float input[5] = { 2.4450693130493164,-0.37077051401138306,1.3057862520217896,-1.2073304653167725,-1.4504870176315308 };
float output[5];
forward(input, output);
for (int i = 0; i < 5; i++){ printf("%f  ", output[i]); 
 }
return 0;
}