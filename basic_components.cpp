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
            printf("%.4f ", mat[i*column+j]);
        }
        printf("\n");
    }
}
// =============== fully-connected layer ========================//
fully_connected::fully_connected(int in_dim, int out_dim)
{
    weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    bias = (float*)malloc(sizeof(float) * out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    _in_dim = in_dim;
    _out_dim = out_dim;
}

fully_connected::fully_connected(int in_dim, int out_dim, float* weight_pretrained, float* bias_pretrained)
{
    weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    bias = (float*)malloc(sizeof(float) * out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
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

// ================= activation function ==================== //
float* ReLU(float* input, int in_size, float* output)
{
    if (output==NULL)
    {
        output = input; // calculate inplace
    }
    for (int i = 0; i < in_size; ++i)
    {
        output[i] = (input[i]>0) ? input[i] : 0.0;
    }
    return output;
}

// ================== convolution layer ====================== //
convolution::convolution(int in_dim, int out_dim, int kernel_size, int stride, int padding):
_in_dim(in_dim),_out_dim(out_dim),_kernel_size(kernel_size), _stride(stride), _padding(padding)
{
    weight = (float**)malloc(sizeof(float*)*out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < out_dim; ++i)
    {
        weight[i] = (float*)malloc(sizeof(float)*kernel_size*kernel_size*in_dim);
        if (weight[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    bias = (float*)malloc(sizeof(float)*out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
}

convolution::~convolution()
{
    for (int i = 0; i < _out_dim; ++i)
    {
        free(weight[i]);
    }
    free(weight);
    free(bias);
}

int convolution::set_weight(float** weight_pretrained, float* bias_pretrained)
{
    for (int i = 0; i < _out_dim; ++i)
    {
        for (int j = 0; j < _kernel_size*_kernel_size*_in_dim; ++j)
        {
            weight[i][j] = weight_pretrained[i][j];
        }
        bias[i] = bias_pretrained[i];
    }
    return 1;
}

float** convolution::forward(float** input, int in_size, float** output)
{
    int in_size_padding = in_size + 2*_padding;
    int out_size = (in_size + 2*_padding - _kernel_size)/_stride + 1;

    //------- padding --------//
    float** input_padding;
    if (_padding == 0) // do not need padding
    {
        input_padding = input;
    }
    else // need padding
    {
        input_padding = (float**)malloc(sizeof(float*)*_in_dim);
        if (input_padding == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
        for (int i = 0; i < _in_dim; ++i)
        {
            input_padding[i] = (float*)malloc(sizeof(float)*in_size_padding*in_size_padding);
            if (input_padding[i] == NULL)
            {
                printf("Malloc failed!\n");
                exit(1);
            }
            for (int j = 0; j < in_size_padding; ++j)
            {
                for (int k = 0; k < in_size_padding; ++k)
                {
                    if ((j < _padding) || (j >= in_size_padding - _padding) || (k < _padding) || (k >= in_size_padding - _padding))
                    {
                        input_padding[i][j*in_size_padding + k] = 0; // zero padding
                    }
                    else
                        input_padding[i][j*in_size_padding + k] = input[i][(j-_padding)*in_size + k - _padding]; // copy initial input value
                }
            }
        }
    }

    //--------calculate convolution--------//
    for (int i = 0; i < _out_dim; ++i)  // output channel iteration
    {
        for (int m = 0; m < out_size; ++m)
        {
            for (int n = 0; n < out_size; ++n)  // output element iteration (in each channel)
            {
                // output(i,m,n) = W(i).*input(:) + bias(i)
                int lt_r = m*_stride; // left top row num
                int lt_c = n*_stride; // left top column num

                output[i][m*out_size+n] = 0;
                for (int s = 0; s < _in_dim; ++s) // input_padding channel iteration
                {
                    for (int t = 0; t < _kernel_size; ++t) // input_padding kernel row iteration
                    {
                        output[i][m*out_size+n] += 
                            cblas_sdot(_kernel_size, &input_padding[s][(lt_r+t)*in_size_padding+lt_c], 1, &weight[i][(s*_kernel_size+t)*_kernel_size], 1);
                    }
                }
                //bias
                output[i][m*out_size+n] += bias[i];
            }
        }
    }

    // -------- free input_padding memory--------//
    if (_padding != 0) // need to free padding memory
    {
        for (int i = 0; i < _in_dim; ++i)
        {
            free(input_padding[i]);
        }
        free(input_padding);
    }
    return output;
}