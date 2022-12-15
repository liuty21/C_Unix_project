#include <stdio.h>
#include "basic_components.h"
#include "Models.h"
#include <stdlib.h>
#include <math.h>
#include "CImg.h"
using namespace std;
using namespace cimg_library;

void show_predict_result(unsigned char* img_in, int model_output)
{
    CImg<unsigned char> img1(img_in,28,28);
    char title_name[] = "model output = 0";
    title_name[15] = title_name[15]+model_output; // show model result in title
    img1.display(title_name, false, 0, true);
}

int main(int argc, char const *argv[])
{
    int id = 1, predict_result = 0; // NOTE: 0<= id < 10000
    FILE* dataset = fopen("../MNIST/t10k-images.idx3-ubyte","rb");
    if (dataset<0)
    {
        printf("Cannot open file\n");
        exit(2);
    }
    unsigned char img[28*28];
    float* img_input = new float[28*28];
    float output[10];
    fseek(dataset, 4*4+id*28*28, SEEK_SET); // move to target image with specified id
    fread(&img, 28*28, 1, dataset);

    for (int i = 0; i < 28*28; ++i) // convert to float type for model input
    {
        img_input[i] = img[i];
    }

    AlexNet model;
    char model_path[] = "../models/model_AlexNet";  // model file path
    model.set_weight(model_path);
    model.forward(&img_input, output, 28);
    // print_matrix(output, 1, 10);

    float output_max = -10000; 
    for (int i = 0; i < 10; ++i) // find argmax(output)
    {
        if (output_max < output[i])
        {
            output_max = output[i];
            predict_result = i;
        } 
    }
    show_predict_result(img, predict_result); // show image


    fclose(dataset);
    delete[]img_input;
    return 0;
}