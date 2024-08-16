#include <stdio.h>
#include <stdlib.h>

// MaxPool1D 函数
void MaxPool1D(int batch_size, int elements_length, int pool_size, int stride, float *input, float *output)
{
    int output_length = (elements_length - pool_size) / stride + 1;

    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < output_length; i++)
        {
            float max_value = input[b * elements_length + i * stride];
            for (int j = 1; j < pool_size; j++)
            {
                float value = input[b * elements_length + i * stride + j];
                if (value > max_value)
                {
                    max_value = value;
                }
            }
            output[b * output_length + i] = max_value;
        }
    }
}

int main()
{
    int batch_size = 2;
    int elements_length = 10;
    int pool_size = 2;
    int stride = 2;

    float input[20] = {
        1, -1, 2, -2, 3, -3, 4, -4, 5, -5,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float output[10];

    // 使用 MaxPool1D
    MaxPool1D(batch_size, elements_length, pool_size, stride, input, output);

    // 输出结果
    printf("C 语言 MaxPool1D 输出结果:\n");
    for (int i = 0; i < batch_size * (elements_length / stride); i++)
    {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}
