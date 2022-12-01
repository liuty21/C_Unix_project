#ifndef BASIC_COMPNENTS
#define BASIC_COMPNENTS
#include <stddef.h>
void print_matrix(float* mat, int row, int column);

class fully_connected
{
public:
    fully_connected(); // default constructor
    fully_connected(int in_dim, int out_dim); // Constructor with dimension information
    fully_connected(const fully_connected& fc); //copy constructor
    ~fully_connected(); // NOTE: remember to free memory
    int set_weight(float* weight_pretrained, float* bias_pretrained);
    float* forward(float* input, float* output); // forward propagation

    int _in_dim;
    int _out_dim;
    float* weight; // weight(size = out_dim x in_dim)
    float* bias; // bias(size = out_dim x 1)
};

class max_pooling
{
public:
    max_pooling(); // default constructor
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
    convolution(); // default constructor
    convolution(int in_dim, int out_dim, int kernel_size, int stride, int padding=0);
    convolution(const convolution& conv); //copy constructor
    ~convolution();
    int set_weight(float** weight_pretrained, float* bias_pretrained);
    float** forward(float** input, int in_size, float** output);

    int _in_dim;
    int _out_dim;
    int _kernel_size;
    int _stride;
    int _padding;
    float** weight; // out_dim x (kernel_size x kernel_size x in_dim)
    float* bias;  // out_dim x 1
    bool valid; // is layer valid (initialized weight) or not
};

class residual
{
public:
    residual(int in_dim, int out_dim, int stride=1); // stride = 1: direct connect; stride = 2: change size (half)
    ~residual();
    int set_weight(float** weight1, float* bias1, float** weight2, float* bias2, float** weight_i, float* bias_i);
    // usually the bias* will be set 0 in resnet
    float** forward(float** input, int in_size, float** output);
private:
    convolution conv1,conv2;
    int _type;
    int _in_dim;
    int _out_dim;
    convolution conv_identity;
};
#endif