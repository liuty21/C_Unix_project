import torch
import numpy as np
def write_txt(state_dict, model_name):
	f = open('../models/'+model_name,'w')
	for layer in state_dict:
		data = state_dict[layer]
		# print(layer, ':',data)
		data = data.view(-1)
		# print(len(data))
		
		for x in range(len(data)): # order: out channel -> in channel -> kernel
			# print(data[x].item())
			f.write(str(data[x].item())+' ')
			
		f.write('\n')
	
	f.close()

def print_model(state_dict):
	for layer in state_dict:
		data = state_dict[layer]
		print(layer, ':',data)
		# print(layer)

def main():
	# model_path = './Model_Alexnet.pth'
	# model_path = './Model_resnet.pth'
	model_path = './Model_Vggnet.pth'

	state_dict = torch.load(model_path)

	# write_txt(state_dict, "model_AlexNet")
	# write_txt(state_dict, "model_ResNet")
	write_txt(state_dict, "model_VGGNet")
	print_model(state_dict)

if __name__ == '__main__':
	main()