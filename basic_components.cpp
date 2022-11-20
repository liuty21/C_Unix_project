#include "basic_components.h"
#include "cblas.h"
#include <stdlib.h>

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