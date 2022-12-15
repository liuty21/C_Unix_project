#include "Models.h"
#include <stdlib.h>
#include <stdio.h>

AlexNet::AlexNet(int dim1, int dim2, int dim3, int kernel_size1, int kernel_size2, int kernel_size3,
        int stride1, int stride2, int stride3, int padding1, int padding2, int padding3,
        int fc_size1, int fc_size2, int fc_size3):
_dim1(dim1), _dim2(dim2), _dim3(dim3), _kernel_size1(kernel_size1), _kernel_size2(kernel_size2), _kernel_size3(kernel_size3), 
_stride1(stride1), _stride2(stride2), _stride3(stride3), _padding1(padding1), _padding2(padding2), _padding3(padding3), 
_fc_size1(fc_size1), _fc_size2(fc_size2), _fc_size3(fc_size3),
conv1(1, dim1, kernel_size1, stride1, padding1),
conv2(dim1, dim2, kernel_size2, stride2, padding2), 
conv3(dim2, dim3, kernel_size3, stride3, padding3),
pool1(2,2), pool2(2,2), 
fc1(fc_size1, fc_size2), fc2(fc_size2, fc_size3)
{
    printf("Model AlexNet constructed\n");
}

AlexNet::~AlexNet(){}

int AlexNet::set_weight(float** kernel1, float* bias_c1, float** kernel2, float* bias_c2,
    float** kernel3, float* bias_c3, float* weight_fc1, float* bias_fc1, float* weight_fc2, float* bias_fc2)
{
    conv1.set_weight(kernel1, bias_c1);
    conv2.set_weight(kernel2, bias_c2);
    conv3.set_weight(kernel3, bias_c3);
    fc1.set_weight(weight_fc1, bias_fc1);
    fc2.set_weight(weight_fc2, bias_fc2);
    return 1;
}

int AlexNet::set_weight(char* model_path)
{
    // -------- open model file -------- //
    FILE* fi = fopen(model_path, "r");
    if (fi<0)
    {
        printf("Cannot open model file: %s\n", model_path);
        exit(2);
    }
    printf("open model file %s\n", model_path);
    float** kernel;
    float* weight;
    float* bias;
    int kernel_len;

    // -------- set conv1 layer parameters -------- //
    // malloc memory
    kernel = (float**)malloc(sizeof(float*)*_dim1);
    kernel_len = 1*_kernel_size1*_kernel_size1;
    if (kernel == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim1; ++i)
    {
        kernel[i] = (float*)malloc(sizeof(float)*kernel_len);
        if (kernel[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }
    bias = (float*)malloc(sizeof(float)*_dim1);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // load parameters from model file
    for (int i = 0; i < _dim1; ++i)
    {
        for (int j = 0; j < kernel_len; ++j)
        {
            fscanf(fi, "%f", &kernel[i][j]); // read in weight values
        }
    }
    for (int i = 0; i < _dim1; ++i)
    {
        fscanf(fi, "%f", bias+i); // read in bias values
    }
    conv1.set_weight(kernel, bias); // set layer weight

    // free memory
    for (int i = 0; i < _dim1; ++i)
    {
        free(kernel[i]);
    }
    free(kernel);
    free(bias);

    printf("conv1 layer initialized\n");

    // -------- set conv2 layer parameters -------- //
    // malloc memory
    kernel = (float**)malloc(sizeof(float*)*_dim2);
    kernel_len = _dim1 * _kernel_size2 * _kernel_size2;
    if (kernel == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim2; ++i)
    {
        kernel[i] = (float*)malloc(sizeof(float)*kernel_len);
        if (kernel[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }
    bias = (float*)malloc(sizeof(float)*_dim2);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // load parameters from model file
    for (int i = 0; i < _dim2; ++i)
    {
        for (int j = 0; j < kernel_len; ++j)
        {
            fscanf(fi, "%f", &kernel[i][j]); // read in weight values
        }
    }
    for (int i = 0; i < _dim2; ++i)
    {
        fscanf(fi, "%f", bias+i); // read in bias values
    }
    conv2.set_weight(kernel, bias); // set layer weight

    // free memory
    for (int i = 0; i < _dim2; ++i)
    {
        free(kernel[i]);
    }
    free(kernel);
    free(bias);

    printf("conv2 layer initialized\n");

    // -------- set conv3 layer parameters -------- //
    // malloc memory
    kernel = (float**)malloc(sizeof(float*)*_dim3);
    kernel_len = _dim2 * _kernel_size3 * _kernel_size3;
    if (kernel == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim3; ++i)
    {
        kernel[i] = (float*)malloc(sizeof(float)*kernel_len);
        if (kernel[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }
    bias = (float*)malloc(sizeof(float)*_dim3);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // load parameters from model file
    for (int i = 0; i < _dim3; ++i)
    {
        for (int j = 0; j < kernel_len; ++j)
        {
            fscanf(fi, "%f", &kernel[i][j]); // read in weight values
        }
    }
    for (int i = 0; i < _dim3; ++i)
    {
        fscanf(fi, "%f", bias+i); // read in bias values
    }
    conv3.set_weight(kernel, bias); // set layer weight

    // free memory
    for (int i = 0; i < _dim3; ++i)
    {
        free(kernel[i]);
    }
    free(kernel);
    free(bias);

    printf("conv3 layer initialized\n");

    // -------- set fc1 layer parameters -------- //
    // malloc memory
    weight = (float*)malloc(sizeof(float*)*_fc_size1*_fc_size2);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    bias = (float*)malloc(sizeof(float)*_fc_size2);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // load parameters from model file
    for (int i = 0; i < _fc_size1*_fc_size2; ++i)
    {
        fscanf(fi, "%f", weight+i); // read in weight values
    }
    for (int i = 0; i < _fc_size2; ++i)
    {
        fscanf(fi, "%f", bias+i); // read in bias values
    }
    fc1.set_weight(weight, bias); // set layer weight

    // free memory
    free(weight);
    free(bias);

    printf("fc1 layer initialized\n");

    // -------- set fc2 layer parameters -------- //
    // malloc memory
    weight = (float*)malloc(sizeof(float*)*_fc_size2*_fc_size3);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    bias = (float*)malloc(sizeof(float)*_fc_size3);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // load parameters from model file
    for (int i = 0; i < _fc_size2*_fc_size3; ++i)
    {
        fscanf(fi, "%f", weight+i); // read in weight values
    }
    for (int i = 0; i < _fc_size3; ++i)
    {
        fscanf(fi, "%f", bias+i); // read in bias values
    }
    fc2.set_weight(weight, bias); // set layer weight

    // free memory
    free(weight);
    free(bias);
    printf("fc2 layer initialized\n");

    fclose(fi);

    return 1;
}

float* AlexNet::forward(float** input, float* output, int in_size)
{
    // -------- calculate intermediate outputs size --------//
    int c1_out_size = output_size_cal(in_size, _kernel_size1, _stride1, _padding1); // conv1 output
    int p1_out_size = output_size_cal(c1_out_size, 2, 2); // pooling1 output
    int c2_out_size = output_size_cal(p1_out_size, _kernel_size2, _stride2, _padding2); // conv2 output
    int p2_out_size = output_size_cal(c2_out_size, 2, 2); // pooling2 output
    int c3_out_size = output_size_cal(p2_out_size, _kernel_size3, _stride3, _padding3); // conv3 output
    
    float** inner_input; // intermediate input pointer
    float** inner_output; // intermediate output pointer
    
    // ----------- layer 1: conv1+ReLU ------------ //
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*_dim1);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim1; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*c1_out_size*c1_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    conv1.forward(input, in_size, inner_output);
    ReLU(inner_output, _dim1, c1_out_size);

    // ----------- layer 2: pool1 ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*_dim1);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim1; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*p1_out_size*p1_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    pool1.forward(inner_input, _dim1, c1_out_size, inner_output);

    // free input memory
    for (int i = 0; i < _dim1; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 3: conv2+ReLU ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*_dim2);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim2; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*c2_out_size*c2_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    conv2.forward(inner_input, p1_out_size, inner_output);
    ReLU(inner_output, _dim2, c2_out_size);

    // free input memory
    for (int i = 0; i < _dim1; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 4: pool2 ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*_dim2);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim2; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*p2_out_size*p2_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    pool2.forward(inner_input, _dim2, c2_out_size, inner_output);

    // free input memory
    for (int i = 0; i < _dim2; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 5: conv3+ReLU ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*_dim3);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _dim3; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*c3_out_size*c3_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    conv3.forward(inner_input, p2_out_size, inner_output);
    ReLU(inner_output, _dim3, c3_out_size);

    // free input memory
    for (int i = 0; i < _dim2; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // --------- convert 2d images to 1d vector -------- //
    float* fc_input; // 1d intermediate input pointer
    float* fc_output;// 1d intermediate output pointer
    fc_input = (float*)malloc(sizeof(float)*_fc_size1); 
    // assert _fc_size1 == _dim3*c3_out_size*c3_out_size
    int index = 0;
    for (int d = 0; d < _dim3; ++d)
    {
        for (int i = 0; i < c3_out_size; ++i)
        {
            for (int j = 0; j < c3_out_size; ++j)
            {
                fc_input[index] = inner_output[d][i*c3_out_size+j];
                ++index;
            }
        }
    }

    // free input memory
    for (int i = 0; i < _dim3; ++i)
    {
        free(inner_output[i]);
    }
    free(inner_output);

    // ---------- layer 6: fc1+ReLU ---------- //
    fc_output = (float*)malloc(sizeof(float)*_fc_size2);
    if (fc_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    // forward calculation
    fc1.forward(fc_input, fc_output);
    ReLU(fc_output, _fc_size2);

    // free input memory
    free(fc_input);

    // -------- layer 7: fc2 --------- //
    fc_input = fc_output;
    fc2.forward(fc_input, output);
    free(fc_input);

    return output;
}