#ifndef model
#define model
float *Linear(int batch_size, int input_size, int output_size, float *input, float *weight_transposed, float *bias);
void Relu(int batch_size, int elements_length, float *input);
void LeakyReLU(int batch_size, int elements_length, float *input, float alpha);
void SoftMax(int batch_size, int elements_length, float *input);
void forward(float input[], float output[]);
#endif
