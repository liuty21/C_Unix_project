#ifndef BASIC_COMPNENTS
#define BASIC_COMPNENTS

class fully_connected
{
public:
	fully_connected(int in_dim, int out_dim);
	fully_connected(int in_dim, int out_dim, float* weight_pretrained, float* bias_pretrained); // Constructor with initialization
	~fully_connected(); // NOTE: remember to free memory
	int set_weight(float* weight_pretrained, float* bias_pretrained);
	float* forward(float* input, float* output); // forward propagation
private:
	int _in_dim;
	int _out_dim;
	float* weight; // weight(size = out_dim x in_dim)
	float* bias; // bias(size = out_dim x 1)
};



#endif