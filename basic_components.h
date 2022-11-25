#ifndef BASIC_COMPNENTS
#define BASIC_COMPNENTS
#include <stddef.h>
void print_matrix(float* mat, int row, int column);

class fully_connected
{
public:
    fully_connected(int in_dim, int out_dim);
    fully_connected(int in_dim, int out_dim, float* weight_pretrained, float* bias_pretrained); // Constructor with initialization
    ~fully_connected(); // NOTE: remember to free memory
    int set_weight(float* weight_pretrained, float* bias_pretrained);
    float* forward(float* input, float* output); // forward propagation
private:
    int _in_dim;
    int _out_dim;
    float* weight; // weight(size = out_dim x in_dim)
    float* bias; // bias(size = out_dim x 1)
};

class max_pooling
{
public:
    max_pooling(int kernel_size, int stride);
    ~max_pooling();
    float* forward(float* input, int in_size, float* output); //forward propagation
private:
    int _kernel_size;
    int _stride;
};

float* ReLU(float* input, int in_size, float* output=NULL);

class convolution
{
public:
    convolution(int in_dim, int out_dim, int kernel_size, int stride, int padding=0);
    ~convolution();
    int set_weight(float** weight_pretrained, float* bias_pretrained);
    float** forward(float** input, int in_size, float** output);
private:
    int _in_dim;
    int _out_dim;
    int _kernel_size;
    int _stride;
    int _padding;
    float** weight; // out_dim x (kernel_size x kernel_size x in_dim)
    float* bias;  // out_dim x 1
};
#endif