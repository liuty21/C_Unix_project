#include <stdio.h>
#include "basic_components.h"
#include "Models.h"
#include <stdlib.h>
#include <math.h>
#include "CImg.h"
using namespace std;
using namespace cimg_library;
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

// // test maxpooling layer & ReLU
// int main(int argc, char const *argv[])
// {
//     int kernel_size = 3, stride = 2;
//     int in_size = 8, in_dim = 2;
//     int out_size = output_size_cal(in_size, kernel_size, stride);
//     float** input;
//     float** output;
//     float input1[64], input2[64];
//     input = new float*[in_dim];
//     output = new float*[in_dim];
//     for (int i = 0; i < in_dim; ++i)
//     {
//         output[i] = new float[out_size*out_size];
//     }
//     input[0]=input1; input[1] = input2;

//     float** activate;
//     for (int i = 0; i < 64; ++i)
//     {
//         input1[i] = i+1;
//         input2[i] = -i+1;
//     }
//     for(int i=0; i < in_dim; ++i)
//     {
//         print_matrix(input[i], in_size, in_size);
//         printf("\n");
//     }
//     max_pooling pool(kernel_size, stride);
//     pool.forward(input, in_dim, in_size, output);

//     for(int i=0; i < in_dim; ++i)
//     {
//         print_matrix(output[i], out_size, out_size);
//         printf("\n");
//     }

//     activate = ReLU(output, in_dim, out_size);
//     for(int i=0; i < in_dim; ++i)
//     {
//         print_matrix(activate[i], out_size, out_size);
//         printf("\n");
//     }

//     for (int i = 0; i < in_dim; ++i)
//     {
//         delete []output[i];
//     }
//     delete []output;
//     delete []input;

//     return 0;
// }


// // test convolution layer
// int main(int argc, char const *argv[])
// {
//     int in_size = 4, out_size = 2;
//     int in_dim = 2, out_dim = 4, kernel_size = 3, stride = 2, padding = 1;
//     convolution conv=convolution(in_dim, out_dim, kernel_size, stride, padding);

//     float **input = new float*[in_dim];
//     for (int i = 0; i < in_dim; ++i)
//     {
//         input[i]= new float[in_size*in_size];
//         for (int j = 0; j < in_size*in_size; ++j)
//         {
//             input[i][j] = j+1;
//         }
//     }
//     float** output = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         output[i] = new float[out_size*out_size];
//     }

//     float** weight = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         weight[i] = new float[kernel_size*kernel_size*in_dim];
//         for (int j = 0; j < kernel_size*kernel_size*in_dim; ++j)
//         {
//             weight[i][j] = 0.1*(i+1);
//         }
//         // printf("weight%d:\n", i+1);
//         // print_matrix(weight[i], kernel_size, kernel_size);
//     }
//     float* bias = new float[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         bias[i] = 1;
//     }

//     conv.set_weight(weight, bias);
//     conv.forward(input, in_size, output);

//     for (int i = 0; i < out_dim; ++i)
//     {
//         print_matrix(output[i], out_size, out_size);
//         printf("\n");
//     }

//     for (int i = 0; i < out_dim; ++i)
//     {
//         delete []weight[i];
//         delete []output[i];
//     }
//     delete []weight;
//     delete []output;
//     delete []input;
//     delete []bias;
//     return 0;
// }

// // test residual block
// int main(int argc, char const *argv[])
// {
//     int in_size = 4;
//     int in_dim = 2, out_dim = 4, kernel_size = 3, stride = 2;
//     int out_size = in_size/stride;

//     residual res(in_dim, out_dim, stride);

//     float **input = (float**)malloc(sizeof(float*)*in_dim);
//     for (int i = 0; i < in_dim; ++i)
//     {
//         input[i]= (float*)malloc(sizeof(float)*in_size*in_size);
//         for (int j = 0; j < in_size*in_size; ++j)
//         {
//             input[i][j] = j+1;
//         }
//     }
//     printf("%f\n", input[0][1]);
//     float** output = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         output[i] = new float[out_size*out_size];
//     }

//     float** weight1 = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         weight1[i] = new float[kernel_size*kernel_size*in_dim];
//         for (int j = 0; j < kernel_size*kernel_size*in_dim; ++j)
//         {
//             weight1[i][j] = 0.1;
//         }
//         // printf("weight%d:\n", i+1);
//         // print_matrix(weight[i], kernel_size, kernel_size);
//     }
//     float** weight2 = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         weight2[i] = new float[kernel_size*kernel_size*out_dim];
//         for (int j = 0; j < kernel_size*kernel_size*out_dim; ++j)
//         {
//             weight2[i][j] = 0.1;
//         }
//         // printf("weight%d:\n", i+1);
//         // print_matrix(weight[i], kernel_size, kernel_size);
//     }

//     float* bias = new float[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         bias[i] = 0;
//     }

//     float** weight_i = new float*[out_dim];
//     for (int i = 0; i < out_dim; ++i)
//     {
//         weight_i[i] = new float[1*1*in_dim];
//         for (int j = 0; j < in_dim; ++j)
//         {
//             weight_i[i][j] = 1;
//         }
//     }

// printf("%f\n", input[0][1]);
//     res.set_weight(weight1, bias, weight2, bias, weight_i, bias);
//     printf("%f\n", input[0][1]);
//     res.forward(input, in_size, output);

//     for (int i = 0; i < out_dim; ++i)
//     {
//         print_matrix(output[i], out_size, out_size);
//         printf("\n");
//     }

//     for (int i = 0; i < in_dim; ++i)
//     {
//         free(input[i]);
//     }
//     free(input);
//     for (int i = 0; i < out_dim; ++i)
//     {
//         delete []weight1[i];
//         delete []weight2[i];
//         delete []output[i];
//     }
//     delete []weight1;
//     delete []weight2;
//     delete []output;
//     delete []bias;
//     return 0;
// }


// // test AlexNet model
// int main(int argc, char const *argv[])
// {
//     char model_path[] = "./model.txt";
//     AlexNet model;
//     printf("create model AlexNet\n");
//     model.set_weight(model_path);
//     float* image = new float[28*28];
//     for (int i = 0; i < 28*28; ++i)
//     {
//         image[i] = 0;
//     }
//     float* output = new float[10];
//     model.forward(&image, output, 28);
//     print_matrix(output, 1, 10);
//     delete[]image;
//     delete[]output;
// }

// load MNIST img
void reverse_endian(int* buffer, int length)
{
    char temp;
    char* swap;
    for (int i = 0; i < length; ++i)
    {
        swap=(char*)(buffer+i);
        temp = swap[0];
        swap[0] = swap[3];
        swap[3] = temp;
        temp = swap[1];
        swap[1] = swap[2];
        swap[2] = temp;
    }
}
// test CImg library and model forward function
int main(int argc, char const *argv[])
{
    int numimg;
    FILE* dataset = fopen("../MNIST/t10k-images.idx3-ubyte","rb");
    if (dataset<0)
    {
        printf("Cannot open file\n");
        exit(2);
    }
    unsigned char img[28*28];
    float* img_input = new float[28*28];
    float output[10];
    fseek(dataset, 4*4, SEEK_SET);
    fread(&img, 28*28, 1, dataset);

    for (int i = 0; i < 28*28; ++i)
    {
        img_input[i] = img[i];
    }

    // print_matrix(img_input, 28,28);

    AlexNet model;
    char model_path[] = "../models/model.txt";
    model.set_weight(model_path);
    model.forward(&img_input, output, 28);
    print_matrix(output, 1, 10);

    CImg<unsigned char> img1(img,28,28);
    img1.display("test");

    fclose(dataset);
    delete[]img_input;
    return 0;
}