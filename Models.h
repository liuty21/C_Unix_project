#ifndef MODELS
#define MODELS
#include "basic_components.h"

// AlexNet-like Network for MNIST dataset
class AlexNet
{
public:
	AlexNet();
	~AlexNet();
	int set_weight(float** kernel1, float* bias_c1, float** kernel2, float* bias_c2, float** kernel3, float* bias_c3, 
		float* weight_fc1, float* bias_fc1, float* weight_fc2, float* bias_fc2);
	float* forward(float** input, float* output, int in_size=28);

	convolution conv1, conv2, conv3; // convolutional layers
	maxpooling pool1, pool2;  // max-pooling layers
	fully_connected fc1, fc2; // fully-connected layers
	
	// assume input size of image = 1
	int dim1, dim2, dim3; // hidden layers' dimension
	int kernel_size1, kernel_size2, kernel_size3;
	int stride1, stride2, stride3;
	int padding1, padding2, padding3;
	int fc_size1, fc_size2, fc_size3;
};


#endif