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
	model_path = './Model_Alexnet.pth'
	state_dict = torch.load(model_path)
	write_txt(state_dict, "model_AlexNet")
	# print_model(state_dict)

if __name__ == '__main__':
	main()