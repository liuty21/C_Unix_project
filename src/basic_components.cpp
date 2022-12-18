#include "basic_components.h"
#include "cblas.h"
#include <stdlib.h>
#include <stdio.h>

// =============== basic functions ========================//
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

int output_size_cal(int in_size, int kernel_size, int stride, int padding)
{
    int out_size;
    out_size = (in_size + 2*padding - kernel_size)/stride + 1;
    return out_size;
}
// =============== fully-connected layer ========================//
fully_connected::fully_connected()
{
    _in_dim = 0;
    _out_dim = 0;
    weight = NULL;
    bias = NULL;
    valid = false;
} // default constructor

fully_connected::fully_connected(int in_dim, int out_dim)
{
    weight = (float*)malloc(sizeof(float) * in_dim * out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed: fc weight\n");
        exit(1);
    }
    bias = (float*)malloc(sizeof(float) * out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed: fc bias\n");
        exit(1);
    }
    _in_dim = in_dim;
    _out_dim = out_dim;
    valid = false;

    // printf("fc(%d, %d) constructed\n", in_dim, out_dim);
}

fully_connected::fully_connected(const fully_connected& fc)
{
    weight = (float*)malloc(sizeof(float) * fc._in_dim * fc._out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    bias = (float*)malloc(sizeof(float) * fc._out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    _in_dim = fc._in_dim;
    _out_dim = fc._out_dim;
    valid = fc.valid;
    if(valid)
    {
        for (int i = 0; i < _in_dim * _out_dim; ++i)
        {
            weight[i] = fc.weight[i]; //copy weight
        }
        for (int i = 0; i < _out_dim; ++i)
        {
            bias[i] = fc.bias[i]; //copy bias
        }
    }
}

fully_connected::~fully_connected()
{
    if(weight!=NULL)
        free(weight); // free weight memory
    if(bias!=NULL)
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
    valid = true;// set the valid signal
    return 1;
}

float* fully_connected::forward(float* input, float* output)
{
    //------- uninitialize detection --------//
    if (valid == false)
    {
        printf("Error! Uninitialized fc layer cannot use forward().\n");
        return NULL;
    }

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
max_pooling::max_pooling(){}

max_pooling::max_pooling(int kernel_size, int stride)
{
    _kernel_size = kernel_size;
    _stride = stride;
}

max_pooling::~max_pooling(){}

float** max_pooling::forward(float** input, int in_dim, int in_size, float** output)
{
    int out_size = (in_size - _kernel_size)/_stride + 1;
    for(int d = 0; d < in_dim; ++d)
    {
        for (int i = 0; i < out_size; ++i)
        {
            for (int j = 0; j < out_size; ++j)
            {
                // calculation rule: 
                // out(i,j) = max[in(i*st,     j*st)  , ..., in(i*st,       j*st+knl-1)]
                //               [      ...                       ...                  ]
                //               [in(i*st+knl-1, j*st), ..., in(i*st+knl-1, j*st+knl-1)]
    
                // set the first element as output
                output[d][i*out_size+j] = input[d][i*_stride*in_size + j*_stride];
                for (int k = 0; k < _kernel_size; ++k)
                {
                    for (int m = 0; m < _kernel_size; ++m)
                    {
                        // if next element in kernel is larger, change output to the larger element
                        if (output[d][i*out_size+j] < input[d][(i*_stride+k)*in_size + j*_stride+m])
                        {
                            output[d][i*out_size+j] = input[d][(i*_stride+k)*in_size + j*_stride+m];
                        }
                    }
                }
                // finish the calculation of out(d, i, j)
            }
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

float** ReLU(float** input, int in_dim, int in_size, float** output)
{
    if (output == NULL)
    {
        output = input; // calculate inplace
    }
    for (int i = 0; i < in_dim; ++i)
    {
        for (int j = 0; j < in_size * in_size; ++j)
        {
            output[i][j] = (input[i][j]>0) ? input[i][j] : 0.0;
        }
    }
    return output;
}

// ================== convolution layer ====================== //
convolution::convolution()
{
    _in_dim = 0;
    _out_dim = 0;
    _kernel_size = 0;
    _stride = 0;
    _padding = 0;
    weight = NULL;
    bias = NULL;
    valid = false;
}

convolution::convolution(int in_dim, int out_dim, int kernel_size, int stride, int padding):
_in_dim(in_dim),_out_dim(out_dim),_kernel_size(kernel_size), _stride(stride), _padding(padding), valid(false)
{
    weight = (float**)malloc(sizeof(float*)*out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed: conv weight\n");
        exit(1);
    }

    for (int i = 0; i < out_dim; ++i)
    {
        weight[i] = (float*)malloc(sizeof(float)*kernel_size*kernel_size*in_dim);
        if (weight[i] == NULL)
        {
            printf("Malloc failed: conv weight%d\n",i);
            exit(1);
        }
    }

    bias = (float*)malloc(sizeof(float)*out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed: conv bias\n");
        exit(1);
    }

    // printf("conv(%d, %d, %d) constructed\n", in_dim, out_dim, kernel_size);
}

convolution::convolution(const convolution& conv) //NOTE: DO NOT copy default object!
{
    _in_dim = conv._in_dim;
    _out_dim = conv._out_dim;
    _kernel_size = conv._kernel_size;
    _stride = conv._stride;
    _padding = conv._padding;
    valid = conv.valid;

    weight = (float**)malloc(sizeof(float*)*_out_dim);
    if (weight == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < _out_dim; ++i)
    {
        weight[i] = (float*)malloc(sizeof(float)*_kernel_size*_kernel_size*_in_dim);
        if (weight[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    bias = (float*)malloc(sizeof(float)*_out_dim);
    if (bias == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    // if weight is valid, copy weight value
    if(valid)
    {
        for (int i = 0; i < _out_dim; ++i)
        {
            for (int j = 0; j < _kernel_size*_kernel_size*_in_dim; ++j)
            {
                weight[i][j] = conv.weight[i][j];
            }
            bias[i] = conv.bias[i];
        }
    }
}

convolution::~convolution()
{
    if(weight!=NULL)
    {    
        for (int i = 0; i < _out_dim; ++i)
        {
            free(weight[i]);
        }
        free(weight);
    }
    if(bias!=NULL)
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
    valid = true; // set the valid signal
    return 1;
}

float** convolution::forward(float** input, int in_size, float** output)
{
    //------- uninitialize detection --------//
    if (valid == false)
    {
        printf("Error! Uninitialized conv layer cannot use forward().\n");
        return NULL;
    }

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
                    {
                        // printf("%d %d %d %d\n",i,j,k, in_size_padding);
                        input_padding[i][j*in_size_padding + k] = input[i][(j-_padding)*in_size + k - _padding]; // copy initial input value
                    }
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

// =============== residual block ================= //
residual::residual(int in_dim, int out_dim, int stride):
_in_dim(in_dim), _out_dim(out_dim), _stride(stride),
conv1(in_dim, out_dim, 3, stride,1), conv2(out_dim, out_dim, 3,1,1),
conv_identity(in_dim, out_dim, 1, stride)
{
    _type = stride!=1 || out_dim!=in_dim; 
    valid = false;
}

residual::~residual(){}

int residual::set_weight(float** weight1, float* bias1, float** weight2, float* bias2, float** weight_i, float* bias_i)
{
    conv1.set_weight(weight1, bias1);
    conv2.set_weight(weight2, bias2);
    if (_type) // need transform
    {
        conv_identity.set_weight(weight_i, bias_i);
    }
    valid = true; // set valid signal
    return 1;
}

float** residual::forward(float** input, int in_size, float** output)
{
    //------- uninitialize detection --------//
    if (valid == false)
    {
        printf("Error! Uninitialized block cannot use forward().\n");
        return NULL;
    }

    float** conv1_output;
    float** conv_i_output;
    // -------- malloc for intermediate output --------//
    conv1_output = (float**)malloc(sizeof(float*)*_out_dim);
    if (conv1_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    int out_size = (in_size - 1)/_stride + 1; // =in_size or in_size/2
    for (int i = 0; i < _out_dim; ++i)
    {
        conv1_output[i] = (float*)malloc(sizeof(float)*out_size*out_size);
        if (conv1_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // ---------- forward dataflow ----------//
    conv1.forward(input, in_size, conv1_output);
    ReLU(conv1_output, _out_dim, out_size);
    conv2.forward(conv1_output, out_size, output);

    if (_type)
    {
        // reuse the memory of intermediate output
        conv_i_output = conv_identity.forward(input, in_size, conv1_output);
    }
    else
        conv_i_output = input; // direct identity connect to input
    
    for (int i = 0; i < _out_dim; ++i)
    {
        for (int j = 0; j < out_size*out_size; ++j)
        {
            // output = conv(x) + identity(x)
            output[i][j] = output[i][j] + conv_i_output[i][j];
        }
    }

    // output = ReLU(output)
    ReLU(output, _out_dim ,out_size);

    // -------- free intermediate output memory -------- //
    for (int i = 0; i < _out_dim; ++i)
    {
        free(conv1_output[i]);
    }
    free(conv1_output);

    return output;
}