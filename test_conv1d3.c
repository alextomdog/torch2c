#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

// ================== Conv1d: conv1 ================== //
// Weight for conv1d: 5x2x3 @out_channels x in_channels x kernel_size;
float conv1_weights[30] = {-0.22399902, -0.36328125, -0.07659912, -0.02471924, -0.34350586, -0.37500000, 0.30590820, -0.39379883, -0.09216309, -0.16442871, -0.28564453, -0.00712204, 0.40063477, -0.02146912, 0.19775391, 0.35620117, -0.23107910, 0.20898438, -0.31127930, -0.19885254, 0.30541992, 0.01406097, -0.02781677, 0.02745056, -0.10321045, -0.27172852, 0.17749023, -0.14208984, 0.01348114, 0.16772461};
// Bias for conv1d: 5 @out_channels;
float conv1_bias[5] = {0.11779785, -0.34912109, -0.05529785, -0.15820312, -0.32446289};

// ================== Conv1d: conv2 ================== //
// Weight for conv1d: 1x5x1 @out_channels x in_channels x kernel_size;
float conv2_weights[5] = {-0.38305664, 0.44287109, 0.21704102, -0.24548340, -0.25756836};
// Bias for conv1d: 1 @out_channels;
float conv2_bias[1] = {0.22692871};

// ================== MaxPool1d: max_pool1d ================== //
// MaxPool1d for layer: max_pool1d;

// ================== Flatten: flatten ================== //
// Flatten for layer: flatten;

// ================== Layer: fc1 ================== //
// Transposed weights for layer: fc1 @2x8;
float fc1_weight_transposed[16] = {0.31347656, 0.23767090, 0.12408447, -0.62353516, -0.69433594, 0.41357422, -0.20043945, 0.35546875, 0.48559570, -0.27368164, 0.12963867, 0.58203125, -0.51513672, -0.29394531, -0.42016602, -0.52539062};
// Biases for layer: fc1 @8;
float fc1_bias_transposed[8] = {0.63232422, -0.65332031, 0.21923828, 0.54638672, -0.29101562, -0.36669922, 0.00517654, -0.30834961};

// ================== Relu: relu1 ================== //
// Relu for layer: relu1;

// ================== Layer: fc2 ================== //
// Transposed weights for layer: fc2 @8x16;
float fc2_weight_transposed[128] = {-0.10821533, 0.25268555, 0.23327637, -0.06146240, 0.06622314, 0.20715332, -0.05075073, -0.12841797, -0.32177734, -0.02395630, -0.18078613, 0.35009766, -0.05056763, -0.32373047, 0.04708862, -0.10382080, -0.10247803, -0.18969727, 0.22656250, -0.00873566, -0.21130371, -0.28564453, -0.11932373, -0.20983887, -0.00724792, 0.30883789, 0.00569916, 0.05899048, -0.14416504, -0.01044464, 0.05822754, -0.23864746, 0.01934814, 0.25097656, 0.01864624, 0.28637695, 0.14978027, 0.23803711, -0.07952881, -0.14001465, -0.06256104, -0.29882812, 0.21105957, -0.12829590, 0.07354736, 0.34716797, -0.09552002, -0.16442871, 0.24291992, 0.05676270, -0.22460938, 0.23913574, -0.34716797, -0.25561523, -0.13964844, -0.19213867, 0.01180267, 0.11798096, -0.25805664, 0.32031250, 0.07775879, 0.19506836, -0.12585449, -0.19152832, 0.20361328, -0.20739746, -0.03277588, 0.25317383, 0.24670410, 0.15917969, -0.06280518, -0.09478760, -0.04388428, 0.33007812, 0.09197998, 0.22180176, -0.24670410, 0.32495117, 0.21032715, 0.32836914, -0.15637207, -0.27392578, -0.09002686, -0.14453125, -0.08502197, -0.16345215, 0.28295898, 0.05111694, 0.11187744, -0.07916260, 0.12384033, 0.06988525, -0.12396240, 0.34790039, -0.17932129, -0.33789062, -0.13781738, -0.17846680, 0.09735107, 0.24645996, 0.25439453, 0.03213501, -0.07604980, 0.26977539, -0.20043945, -0.13293457, 0.34228516, -0.01096344, -0.08056641, -0.30200195, 0.20727539, -0.01651001, -0.15039062, 0.33007812, -0.06500244, -0.08715820, -0.21948242, 0.29467773, 0.02319336, 0.14306641, 0.09771729, 0.06051636, -0.02973938, -0.25610352, -0.28466797, 0.00722885, -0.32983398, 0.24816895};
// Biases for layer: fc2 @16;
float fc2_bias_transposed[16] = {-0.15295410, -0.03280640, -0.00454712, 0.19116211, -0.00738525, 0.27368164, 0.17932129, 0.25683594, -0.07910156, 0.12866211, -0.13098145, 0.11694336, -0.27343750, 0.03860474, 0.27758789, 0.10980225};

// ================== Layer: fc3 ================== //
// Transposed weights for layer: fc3 @16x10;
float fc3_weight_transposed[160] = {-0.10577393, 0.13159180, 0.18261719, -0.14575195, 0.10052490, 0.08648682, 0.21789551, -0.05117798, 0.00758362, -0.10345459, 0.16210938, -0.20104980, 0.12438965, -0.10412598, 0.04058838, 0.01003265, -0.13867188, 0.18176270, 0.06420898, -0.06475830, 0.00489044, 0.05749512, -0.20727539, 0.00092125, 0.23950195, -0.10845947, -0.18347168, -0.07476807, -0.20520020, -0.04714966, -0.21313477, 0.17883301, -0.18432617, -0.07434082, 0.08581543, -0.04977417, -0.02984619, -0.14794922, 0.16528320, 0.09930420, -0.05419922, -0.19750977, 0.11926270, 0.06927490, 0.11315918, -0.04638672, -0.07147217, 0.07489014, 0.19592285, 0.24548340, -0.22229004, 0.03283691, 0.18945312, -0.04431152, 0.02072144, -0.15710449, 0.11077881, -0.00606918, 0.06481934, -0.18432617, -0.00641632, 0.19384766, -0.00863647, -0.13598633, 0.21923828, -0.18664551, -0.00121689, 0.19848633, -0.02294922, 0.24536133, -0.23303223, -0.01971436, 0.06250000, 0.23242188, -0.08618164, -0.15588379, -0.11071777, -0.07098389, -0.20019531, 0.03350830, -0.20739746, -0.16687012, -0.18811035, -0.13732910, 0.09130859, -0.13439941, 0.16271973, 0.17578125, -0.05456543, -0.13012695, -0.12261963, 0.23474121, -0.08862305, 0.24890137, 0.02479553, -0.14208984, 0.08673096, 0.14013672, -0.14660645, 0.10607910, -0.01870728, -0.18505859, -0.04556274, 0.00227928, 0.22082520, -0.17272949, 0.00594330, 0.17846680, -0.04537964, 0.01472473, 0.12915039, -0.21093750, -0.08129883, -0.10528564, 0.02407837, 0.17065430, 0.19311523, -0.15734863, 0.05456543, -0.02706909, -0.07806396, -0.20666504, 0.03027344, 0.24584961, -0.13745117, -0.13598633, -0.19738770, 0.05969238, -0.07934570, -0.21166992, 0.13720703, 0.14636230, -0.15869141, 0.10510254, 0.03076172, 0.00158024, -0.10852051, -0.16955566, -0.18176270, 0.17712402, -0.14562988, -0.14257812, -0.12768555, 0.15954590, 0.14978027, -0.04202271, 0.03100586, -0.02526855, -0.22998047, 0.21740723, 0.01327515, 0.05569458, -0.13183594, -0.24145508, 0.19543457, -0.23217773, -0.20874023, 0.15966797, 0.18322754, 0.07702637};
// Biases for layer: fc3 @10;
float fc3_bias_transposed[10] = {0.06652832, 0.18798828, 0.02287292, 0.02534485, 0.23315430, 0.23205566, 0.03338623, -0.15246582, 0.04879761, 0.00176907};

// ================== SoftMax: softmax ================== //
// SoftMax for layer: softmax;

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

float* forward(float input[], float output[]){
	float* result_0=(float*)malloc(sizeof(float)*48);
for (int i = 0; i < 48; i++) { result_0[i] = input[i]; }
// conv1_layer
// conv1_layer(batch_size: 1, in_channels: 2, sequence_length: 24)
// => @(batch_size: 1, out_channels: 5, out_sequence_length: 8)
float* result_1 = Conv1d(1,2,24, result_0, 5, 3, 3, 0, conv1_weights, conv1_bias, true);
free(result_0);
// conv2_layer
// conv2_layer(batch_size: 1, in_channels: 5, sequence_length: 8)
// => @(batch_size: 1, out_channels: 1, out_sequence_length: 4)
float* result_2 = Conv1d(1,5,8, result_1, 1, 1, 2, 0, conv2_weights, conv2_bias, true);
free(result_1);
// max_pool1d_layer
// output_size: 2
float* result_3 = MaxPool1D(1,4,2,2,0,result_2);
free(result_2);
// flatten_layer
// fc1_layer
float* result_4 = Linear(1,2,8,result_3,fc1_weight_transposed,fc1_bias_transposed);
free(result_3);
// relu1_relu
Relu(1,8,result_4);
// fc2_layer
float* result_5 = Linear(1,8,16,result_4,fc2_weight_transposed,fc2_bias_transposed);
free(result_4);
// fc3_layer
float* result_6 = Linear(1,16,10,result_5,fc3_weight_transposed,fc3_bias_transposed);
free(result_5);
// softmax_layer
SoftMax(1,10,result_6);
for (int i = 0; i < 10; i++) { output[i] = result_6[i]; }
	free(result_6);
}
int main(){
float input[48] = { -0.23791494965553284,0.23763269186019897,-0.6198317408561707,1.4981762170791626,0.003994853235781193,0.49740907549858093,0.09029236435890198,1.0334904193878174,-0.7652565836906433,0.09394094347953796,-2.214366912841797,-0.7652496099472046,-2.036895990371704,0.5894051194190979,-0.41797080636024475,-1.4406836032867432,-0.39808225631713867,-0.032116055488586426,1.3585180044174194,0.0026323627680540085,-1.8736284971237183,1.0434926748275757,0.49972763657569885,0.11931973695755005,-0.6631355285644531,-0.3757646977901459,-1.5854578018188477,0.8035690188407898,0.8473976254463196,-1.6275094747543335,0.2405077964067459,1.0714585781097412,-0.8047605752944946,0.08846838772296906,0.30354276299476624,-2.409376621246338,1.797454833984375,-1.9932900667190552,1.0923129320144653,0.8093889355659485,0.14190901815891266,-1.0992578268051147,0.402500182390213,2.2180066108703613,0.7015918493270874,1.6954145431518555,-2.8076252937316895,-1.649429440498352 };
float output[10];
forward(input, output);
for (int i = 0; i < 10; i++){ printf("%f  ", output[i]); 
 }
return 0;
}