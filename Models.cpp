#include "Models.h"

AlexNet::AlexNet():
dim1(4), dim2(8), dim3(4), kernel_size1(5), kernel_size2(5), 
kernel_size3(3), stride1(2), stride2(1), stride3(1), padding1(0), 
padding2(2), padding3(1), fc_size1(36), fc_size2(18), fc_size3(10),
conv1(1, dim1, kernel_size1, stride1, padding1),
conv2(dim1, dim2, kernel_size2, stride2, padding2), 
conv3(dim2, dim3, kernel_size3, stride3, padding3),
pool1(2,2), pool2(2,2), 
fc1(fc_size1, fc_size2), fc2(fc_size2, fc_size3)
{}

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

float* AlexNet::forward(float** input, float* output, int in_size)
{
    // -------- calculate intermediate outputs size --------//
    int c1_out_size = output_size_cal(in_size, kernel_size1, stride1, padding1); // conv1 output
    int p1_out_size = output_size_cal(c1_out_size, 2, 2); // pooling1 output
    int c2_out_size = output_size_cal(p1_out_size, kernel_size2, stride2, padding2); // conv2 output
    int p2_out_size = output_size_cal(c2_out_size, 2, 2); // pooling2 output
    int c3_out_size = output_size_cal(p2_out_size, kernel_size3, stride3, padding3); // conv3 output
    
    float** inner_input; // intermediate input pointer
    float** inner_output; // intermediate output pointer
    
    // ----------- layer 1: conv1+ReLU ------------ //
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*dim1);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < dim1; ++i)
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
    ReLU(inner_output, dim1, c1_out_size);

    // ----------- layer 2: pool1 ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*dim1);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < dim1; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*p1_out_size*p1_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    pool1.forward(inner_input, dim1, c1_out_size, inner_output);

    // free input memory
    for (int i = 0; i < dim1; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 3: conv2+ReLU ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*dim2);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < dim2; ++i)
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
    ReLU(inner_output, dim2, c2_out_size);

    // free input memory
    for (int i = 0; i < dim1; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 4: pool2 ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*dim2);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < dim2; ++i)
    {
        inner_output[i] = (float*)malloc(sizeof(float)*p2_out_size*p2_out_size);
        if (inner_output[i] == NULL)
        {
            printf("Malloc failed!\n");
            exit(1);
        }
    }

    // forward calculation
    pool2.forward(inner_input, dim2, c2_out_size, inner_output);

    // free input memory
    for (int i = 0; i < dim2; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // ----------- layer 5: conv3+ReLU ------------ //
    inner_input = inner_output; // next layer's input = last layer's output
    // malloc memory for output
    inner_output = (float**)malloc(sizeof(float*)*dim3);
    if (inner_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }

    for (int i = 0; i < dim3; ++i)
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
    ReLU(inner_output, dim3, c3_out_size);

    // free input memory
    for (int i = 0; i < dim2; ++i)
    {
        free(inner_input[i]);
    }
    free(inner_input);

    // --------- convert 2d images to 1d vector -------- //
    float* fc_input; // 1d intermediate input pointer
    float* fc_output;// 1d intermediate output pointer
    fc_input = (float*)malloc(sizeof(float)*fc_size1); 
    // assert fc_size1 == dim3*c3_out_size*c3_out_size
    int index = 0;
    for (int d = 0; d < dim3; ++d)
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
    for (int i = 0; i < dim3; ++i)
    {
        free(inner_output[i]);
    }
    free(inner_output);

    // ---------- layer 6: fc1+ReLU ---------- //
    fc_output = (float*)malloc(sizeof(float)*fc_size2);
    if (fc_output == NULL)
    {
        printf("Malloc failed!\n");
        exit(1);
    }
    // forward calculation
    fc1.forward(fc_input, fc_output);
    ReLU(fc_output, fc_size2);

    // free input memory
    free(fc_input);

    // -------- layer 7: fc2 --------- //
    fc_input = fc_output;
    fc2.forward(fc_input, output);
    free(fc_input);

    return output;
}