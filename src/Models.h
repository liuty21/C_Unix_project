#ifndef MODELS
#define MODELS
#include "basic_components.h"

// AlexNet-like Network for MNIST dataset
class AlexNet
{
public:
    AlexNet(int dim1=4, int dim2=8, int dim3=4, int kernel_size1=5, int kernel_size2=5, int kernel_size3=3,
        int stride1=2, int stride2=1, int stride3=1, int padding1=0, int padding2=2, int padding3=1,
        int fc_size1=36, int fc_size2=18, int fc_size3=10);
    ~AlexNet();
    int set_weight(float** kernel1, float* bias_c1, float** kernel2, float* bias_c2, float** kernel3, float* bias_c3, 
        float* weight_fc1, float* bias_fc1, float* weight_fc2, float* bias_fc2);
    int set_weight(char* model_path);
    float* forward(float** input, float* output, int in_size=28);

    convolution conv1, conv2, conv3; // convolutional layers
    max_pooling pool1, pool2;  // max-pooling layers
    fully_connected fc1, fc2; // fully-connected layers
    
    // assume input size of image = 1
    int _dim1, _dim2, _dim3; // hidden layers' dimension
    int _kernel_size1, _kernel_size2, _kernel_size3;
    int _stride1, _stride2, _stride3;
    int _padding1, _padding2, _padding3;
    int _fc_size1, _fc_size2, _fc_size3;
};

// ResNet-like Network for MNIST dataset
class ResNet
{
public:
    ResNet(int dim1=4, int dim2=4, int dim3=8, int kernel_size1=5, 
    int stride1=2, int stride2=1, int stride3=2, int padding1=0, 
    int fc_size1=72, int fc_size2=36, int fc_size3=10);
    ~ResNet();

    int set_weight(char* model_path);
    float* forward(float** input, float* output, int in_size=28);

    convolution conv; // input layer
    max_pooling pool;
    residual res1, res2; // residual blocks
    fully_connected fc1, fc2; // fully-connected layers

    // assume input size of image = 1
    int _dim1, _dim2, _dim3; // hidden layers' dimension
    int _kernel_size1, _padding1; // input layer's settings
    int _stride1, _stride2, _stride3;
    int _fc_size1, _fc_size2, _fc_size3;
};

#endif