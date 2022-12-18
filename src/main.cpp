#include <stdio.h>
#include "basic_components.h"
#include "Models.h"
#include <stdlib.h>
#include <math.h>
#include "CImg.h"
using namespace std;
using namespace cimg_library;

// show input img & model output result
void show_predict_result(unsigned char* img_in, int model_output, unsigned char label)
{
    CImg<unsigned char> img1(img_in,28,28);
    char title_name[] = "model output = 0 ( label = 0 )";
    title_name[15] = title_name[15]+model_output; // show model result in title
    title_name[27] = title_name[27]+label;  // show image label in title
    printf("The number is %d\n", model_output);
    img1.display(title_name, false, 0, true);
}

int main(int argc, char const *argv[])
{
    char* model_path;
    AlexNet model_AlexNet;
    ResNet model_ResNet;
    VGGNet model_VGGNet;
    switch(argv[1][0])
    {
        case '0':
            model_path = "../models/model_AlexNet";  // model file path
            model_AlexNet.set_weight(model_path);
            printf("using AlexNet model...\n");
            break;
        case '1':
            model_path = "../models/model_ResNet";  // model file path
            model_ResNet.set_weight(model_path);
            printf("using ResNet model...\n");
            break;
        case '2':
            model_path = "../models/model_VGGNet";  // model file path
            model_VGGNet.set_weight(model_path);
            printf("using VGGNet model...\n");
            break;
        default:
            printf("Error!\n");
            return 1;
    }
    

    // open MNIST test dataset
    FILE* dataset = fopen("../MNIST/t10k-images.idx3-ubyte","rb");
    if (dataset<0)
    {
        printf("Cannot open file\n");
        exit(2);
    }
    // open MNIST test label file
    FILE* datalabel = fopen("../MNIST/t10k-labels.idx1-ubyte","rb");
    if (datalabel<0)
    {
        printf("Cannot open file\n");
        exit(2);
    }

    int id = 0, predict_result = 0;
    unsigned char img[28*28];
    unsigned char label;
    float* img_input = new float[28*28];
    float output[10];
    for (id = 0; id<10; ++id) // NOTE: 0<= id < 10000
    {
        fseek(dataset, 4*4+id*28*28, SEEK_SET); // move to target image with specified id
        fread(&img, 28*28, 1, dataset); // load image

        for (int i = 0; i < 28*28; ++i) // convert to float type for model input
        {
            img_input[i] = img[i];
        }

        // get model predict result
        switch(argv[1][0])
        {
            case '0':
                model_AlexNet.forward(&img_input, output, 28);
                break;
            case '1':
                model_ResNet.forward(&img_input, output, 28);
                break;
            case '2':
                model_VGGNet.forward(&img_input, output, 28);
        }

        float output_max = -10000; 
        for (int i = 0; i < 10; ++i) // find argmax(output)
        {
            if (output_max < output[i])
            {
                output_max = output[i];
                predict_result = i;
            } 
        }

        fseek(datalabel, 2*4+id, SEEK_SET); // move to target label
        fread(&label, 1, 1, datalabel);
        show_predict_result(img, predict_result, label); // show image and result
    }

    fclose(dataset);
    delete[]img_input;
    return 0;
}