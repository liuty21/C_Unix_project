#include <stdio.h>
#include "basic_components.h"

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

// test maxpooling layer
int main(int argc, char const *argv[])
{
    int kernel_size = 3, stride = 2;
    int in_size = 8;
    float input[64], output[16];
    for (int i = 0; i < 64; ++i)
    {
        input[i] = i+1;
    }
    max_pooling pool(kernel_size, stride);
    pool.forward(input, in_size, output);

    print_matrix(output, 3, 3);

    return 0;
}