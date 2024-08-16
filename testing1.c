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

void BatchNorm1d(int batch_size, int elements_length, float *input, float epsilon, float *gamma, float *beta)
{
    float mean[elements_length];
    float variance[elements_length];

    // 1. 计算每个特征的均值
    for (int i = 0; i < elements_length; ++i)
    {
        mean[i] = 0.0f;
        for (int j = 0; j < batch_size; ++j)
        {
            mean[i] += input[j * elements_length + i];
        }
        mean[i] /= batch_size;
    }

    // 2. 计算每个特征的方差
    for (int i = 0; i < elements_length; ++i)
    {
        variance[i] = 0.0f;
        for (int j = 0; j < batch_size; ++j)
        {
            float diff = input[j * elements_length + i] - mean[i];
            variance[i] += diff * diff;
        }
        variance[i] /= batch_size;
    }

    // 3. 对输入进行归一化，并应用gamma和beta
    for (int i = 0; i < elements_length; ++i)
    {
        for (int j = 0; j < batch_size; ++j)
        {
            int idx = j * elements_length + i;
            input[idx] = gamma[i] * ((input[idx] - mean[i]) / sqrt(variance[i] + epsilon)) + beta[i];
        }
    }
}

void forward(float input[], float output[], int length)
{
    float *result_0 = (float *)malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++)
    {
        result_0[i] = input[i];
    }
    // batch_normal_1d_layer
    BatchNorm1d(1, length, result_0, batch_normal_1d_batch_normal1d_epsilon, batch_normal_1d_batch_normal1d_gamma, batch_normal_1d_batch_normal1d_beta);
    for (int i = 0; i < length; i++)
    {
        output[i] = result_0[i];
    }
    free(result_0);
}

int main()
{
    float input[5] = {-0.5959482192993164, 0.40505945682525635, -0.4042966365814209, 0.27326056361198425, 0.06846138834953308};
    float output[5];
    forward(input, output, 5);
    for (int i = 0; i < 5; i++)
    {
        printf("%f  ", output[i]);
    }
    return 0;
}
