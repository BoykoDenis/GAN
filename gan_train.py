import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#import pickle
from PIL import Image
from Gan_model import Generator, Descriminator
import random


lr = 0.001

epochs = 20

flag = False

load_m = False

channels_noise = 1800

batch_size = 4

n_featurs = 30

input_chanels = 3

dataset_size = 100
dataset_move = random.randint(1, 7000)


mod_gen_path = "models\\model_gan_gen1.pth.tar"
mod_des_path = "models\\model_gan_des1.pth.tar"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_noise = torch.randn(1, channels_noise, 1, 1).to(device)



def im_convert(tensor):

	image = tensor.cpu().clone().detach().numpy()

	#clone tensor --> detach it from computations --> transform to numpy

	image = image.squeeze()

	#image = image.transpose(1, 2, 0)

	# swap axis from(1,28,28) --> (28,28,1)

	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

	#denormalize image

	image = image.clip(0, 1)

	#sets image range from 0 to 1

	return image*255





def save_mod(state, filename):

	torch.save(state, filename)

def load_gen_mod(checkpoint):

	gen.load_state_dict(checkpoint['state_dictionary'])

def load_des_mod(checkpoint): 
	
	des.load_state_dict(checkpoint['state_dictionary'])


#training_path = ""
training_path = 'D:\\Datasets\\Mountain\\mountain\\'

training_dataset_raw = [Image.open(training_path+"train (" + str(i*19) + ").jpg") for i in range(1, dataset_size)]

transform_train = transforms.Compose([transforms.Resize((360, 640)),
									  transforms.ToTensor(),

									  #transforms.Normalize((0.5, 0.5, 0.5),

										#				   (0.5, 0.5, 0.5)),

									  ])

for idx, img in enumerate(training_dataset_raw):

	training_dataset_raw[idx] = transform_train(img)

	if idx%100 == 0:

		print(idx, " images have been converted", end = "\r")



training_dataset = torch.stack(training_dataset_raw)
label_setA = torch.ones(dataset_size-1, 1)*0.9
label_setB = torch.ones(dataset_size-1, 1)*0.1
label_set = torch.cat([label_setA, label_setB], dim = 1)
#print(label_set)
#print(label_set.cpu().clone().detach().numpy())
#print(type(label_set),  "  ", label_set.shape)
#print(type(training_dataset),  "  ", training_dataset.shape)
training_dataset_raw = None 
#print(type(training_dataset))
#print(training_dataset.shape)
#print(label_set.shape)
training_dataset = torch.utils.data.TensorDataset(training_dataset, label_set)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
training_dataset = None

#torch.reshape(training_dataset, (-1, batch_size))
torch.cuda.empty_cache()
#sys.exit()
	
criterion = nn.CrossEntropyLoss(reduction = 'sum')
gen = Generator(3, n_featurs).to(device)
des = Descriminator(3, n_featurs).to(device)
gen_parameters = gen.parameters()
des_parameters = des.parameters()
gen_optimizer = optim.Adam(gen_parameters, lr = lr)
des_optimizer = optim.Adam(des_parameters, lr = lr)
graph_gen = []
graph_des = []


if load_m:

	load_gen_mod(torch.load(mod_gen_path))
	load_des_mod(torch.load(mod_des_path))

for epoch in range(epochs):
	#torch.cuda.empty_cache()
	checkpoint_gen = {"state_dictionary": gen.state_dict(), "optimizer": gen_optimizer.state_dict()}
	checkpoint_des = {"state_dictionary": des.state_dict(), "optimizer": des_optimizer.state_dict()}
	running_loss = 0.0
	running_loss_gen = 0.0
	running_loss_des = 0.0


	if epoch % 1 == 0:

		save_mod(checkpoint_gen, filename = "models\\model_gan_gen1.pth.tar")
		save_mod(checkpoint_des, filename = "models\\model_gan_des1.pth.tar")

	for idx, [data, label] in enumerate(training_loader):


		data = data.to(device)
		label = label.to(device)

		fixed_noise = torch.randn(data.shape[0], n_featurs*10, 1, 1).to(device)
		gen_output = gen(fixed_noise).to(device)

		des_out = des(data, n_featurs)
		#print(torch.max(label, 1)[1], "++++ here ++++")
		
		des_loss_true = criterion(des_out, torch.max(label, 1)[1])

		des_out = des(gen_output.detach(), n_featurs)
		#print(des_out.clone().detach().cpu().numpy())
		label_setA = torch.ones(data.shape[0], 1)*0.9
		label_setB = torch.ones(data.shape[0], 1)*0.1
		
		des_label = torch.cat([label_setB, label_setA], dim = 1)
		gen_label = torch.cat([label_setA, label_setB], dim = 1)
		
		des_loss_fake = criterion(des_out, torch.max(des_label, 1)[1].to(device = device, dtype = torch.int64))

		des_loss = (des_loss_fake + des_loss_true)/2

		des_optimizer.zero_grad()
		des_loss.backward()
		des_optimizer.step()

		fixed_noise = torch.randn(data.shape[0], n_featurs*10, 1, 1).to(device)
		gen_output = gen(fixed_noise).to(device)
		des_out = des(gen_output, n_featurs).to(device)

		gen_loss = criterion(des_out, torch.max(gen_label, 1)[1].to(device = device, dtype = torch.int64))
		#criterion(des_out, torch.max(torch.ones(data.shape[0], 1), 1)[1].to(device = device, dtype = torch.int64))

		gen_optimizer.zero_grad()
		gen_loss.backward()
		gen_optimizer.step()
		
		

		running_loss_gen += gen_loss.item()
		running_loss_des += (des_loss_fake + des_loss_true).item()

		graph_gen.append(running_loss_gen/(idx+1))
		graph_des.append(running_loss_des/(idx+1))

		print("epoch: ", epoch, " dataset progress: ", idx, " loss_gen: ", running_loss_gen/(idx+1), " loss_des: ", running_loss_des/(idx+1), end = "\r")


plt.plot(range(1, len(graph_gen)+1), graph_gen)
plt.plot(range(1, len(graph_des)+1), graph_des)
plt.show()



