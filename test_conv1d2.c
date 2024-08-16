#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Conv1d: conv1 ================== //
// Weight for conv1d: 2x2x3 @out_channels x in_channels x kernel_size;
float conv1_weights[12] = {-0.15039062, 0.36816406, -0.23144531, -0.22534180, 0.12170410, -0.25073242, 0.29809570, -0.15161133, -0.14880371, -0.27758789, -0.40576172, -0.05242920};
// Bias for conv1d: 2 @out_channels;
float conv1_bias[2] = {-0.08837891, 0.03811646};

// ================== Flatten: fl1 ================== //
// Flatten for layer: fl1;

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*8);
for (int i = 0; i < 8; i++) { result_0[i] = input[i]; }
// conv1_layer
// conv1_layer(batch_size: 1, in_channels: 2, sequence_length: 4)
// => @(batch_size: 1, out_channels: 2, out_sequence_length: 1)
float* result_1 = Conv1d(1,2,4, result_0, 2, 3, 3, 0, conv1_weights, conv1_bias, true);
free(result_0);
// fl1_layer
for (int i = 0; i < 2; i++) { output[i] = result_1[i]; }
	free(result_1);
}
int main(){
float input[8] = { -0.3213287889957428,1.7627936601638794,-1.2281277179718018,0.6883735656738281,0.27343711256980896,-1.9078199863433838,-0.10110742598772049,-0.9338371753692627 };
float output[2];
forward(input, output);
for (int i = 0; i < 2; i++){ printf("%f  ", output[i]); 
 }
return 0;
}