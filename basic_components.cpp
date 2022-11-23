#include "basic_components.h"
#include "cblas.h"
#include <stdlib.h>
#include <stdio.h>

// =============== print matrix function ========================//
void print_matrix(float* mat, int row, int column)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < column; ++j)
        {
            printf("%.4f\t", mat[i*column+j]);
        }
        printf("\n");
    }
}
// =============== fully-connected layer ========================//
fully_connected::fully_connected(int in_dim, int out_dim)
{
    weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    bias = (float*)malloc(sizeof(float) * out_dim);
    _in_dim = in_dim;
    _out_dim = out_dim;
}

fully_connected::fully_connected(int in_dim, int out_dim, float* weight_pretrained, float* bias_pretrained)
{
    weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    bias = (float*)malloc(sizeof(float) * out_dim);
    _in_dim = in_dim;
    _out_dim = out_dim;

    for (int i = 0; i < _in_dim * _out_dim; ++i)
    {
        weight[i] = weight_pretrained[i]; //copy weight
    }
    for (int i = 0; i < _out_dim; ++i)
    {
        bias[i] = bias_pretrained[i]; //copy bias
    }
}

fully_connected::~fully_connected()
{
    free(weight); // free weight memory
    free(bias); // free bias memory
}

int fully_connected::set_weight(float* weight_pretrained, float* bias_pretrained)
{
    for (int i = 0; i < _in_dim * _out_dim; ++i)
    {
        weight[i] = weight_pretrained[i]; //copy weight
    }
    for (int i = 0; i < _out_dim; ++i)
    {
        bias[i] = bias_pretrained[i]; //copy bias
    }
    return 1;
}

float* fully_connected::forward(float* input, float* output)
{
    const enum CBLAS_ORDER order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    const float alpha = 1;
    const int lda = _in_dim;
    const float beta = 1;
    const int incXY = 1; // to be confirmed

    // copy bias into output first, in order to suit the calculating rule in blas lib
    for (int i = 0; i < _out_dim; ++i)
    {
        output[i] = bias[i];
    }

    // output = weight * input + bias
    cblas_sgemv(order, trans, _out_dim, _in_dim, alpha, weight, lda, input, incXY, beta, output, incXY);
    return output;
}

// ================= max-pooling layer ==================== //
max_pooling::max_pooling(int kernel_size, int stride)
{
    _kernel_size = kernel_size;
    _stride = stride;
}

max_pooling::~max_pooling(){}

float* max_pooling::forward(float* input, int in_size, float* output)
{
    int out_size = (in_size - _kernel_size)/_stride + 1;
    for (int i = 0; i < out_size; ++i)
    {
        for (int j = 0; j < out_size; ++j)
        {
            // calculation rule: 
            // out(i,j) = max[in(i*st,     j*st)  , ..., in(i*st,       j*st+knl-1)]
            //               [      ...                       ...                  ]
            //               [in(i*st+knl-1, j*st), ..., in(i*st+knl-1, j*st+knl-1)]

            // set the first element as output
            output[i*out_size+j] = input[i*_stride*in_size + j*_stride];
            for (int k = 0; k < _kernel_size; ++k)
            {
                for (int m = 0; m < _kernel_size; ++m)
                {
                    // if next element in kernel is larger, change output to the larger element
                    if (output[i*out_size+j] < input[(i*_stride+k)*in_size + j*_stride+m])
                    {
                        output[i*out_size+j] = input[(i*_stride+k)*in_size + j*_stride+m];
                    }
                }
            }
            // finish the calculation of out(i, j)
        }
    }
    return output;
}