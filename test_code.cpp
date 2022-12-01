#include <stdio.h>
#include "basic_components.h"
#include <stdlib.h>
// test fc layer
// int main(int argc, char const *argv[])
// {
//     int in_dim = 4, out_dim = 2;
//     float weight[8] = {1,2,3,4,
//                        5,6,7,8};
//     float bias[2] = {1,1};
//     float input[4] = {1,1,1,1};
//     float output[2];
//     fully_connected fc(in_dim, out_dim, weight, bias);
//     fc.forward(input,output);
//     printf("%f,%f\n", output[0],output[1]);
//     return 0;
// }

// test maxpooling layer & ReLU
// int main(int argc, char const *argv[])
// {
//     int kernel_size = 3, stride = 2;
//     int in_size = 8;
//     float input[64], output[16];
//     float* activate;
//     for (int i = 0; i < 64; ++i)
//     {
//         input[i] = -i+1;
//     }
//     max_pooling pool(kernel_size, stride);
//     pool.forward(input, in_size, output);

//     print_matrix(output, 3, 3);

//     activate = ReLU(output, 9);
//     print_matrix(activate, 3, 3);

//     return 0;
// }


// test convolution layer
int main(int argc, char const *argv[])
{
    int in_size = 4, out_size = 2;
    int in_dim = 2, out_dim = 2, kernel_size = 3, stride = 2, padding = 1;
    convolution conv=convolution(in_dim, out_dim, kernel_size, stride, padding);

    float **input = new float*[in_dim];
    for (int i = 0; i < in_dim; ++i)
    {
        input[i]= new float[in_size*in_size];
        for (int j = 0; j < in_size*in_size; ++j)
        {
            input[i][j] = j+1;
        }
    }
    float** output = new float*[out_dim];
    for (int i = 0; i < out_dim; ++i)
    {
        output[i] = new float[out_size*out_size];
    }

    float** weight = new float*[out_dim];
    for (int i = 0; i < out_dim; ++i)
    {
        weight[i] = new float[kernel_size*kernel_size*in_dim];
        for (int j = 0; j < kernel_size*kernel_size*in_dim; ++j)
        {
            weight[i][j] = 0.1*(i+1);
        }
        // printf("weight%d:\n", i+1);
        // print_matrix(weight[i], kernel_size, kernel_size);
    }
    float* bias = new float[out_dim];
    for (int i = 0; i < out_dim; ++i)
    {
        bias[i] = 1;
    }

    conv.set_weight(weight, bias);
    conv.forward(input, in_size, output);

    for (int i = 0; i < out_dim; ++i)
    {
        print_matrix(output[i], out_size, out_size);
        printf("\n");
    }

    for (int i = 0; i < out_dim; ++i)
    {
        delete []weight[i];
        delete []output[i];
    }
    delete []weight;
    delete []output;
    delete []input;
    delete []bias;
    return 0;
}