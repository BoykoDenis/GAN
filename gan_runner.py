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

n_featurs = 30

input_chanels = 3

channels_noise = 300

dataset_size = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator(3, n_featurs).to(device)

def load_gen_mod(checkpoint):

	gen.load_state_dict(checkpoint['state_dictionary'])

def im_convert(tensor):
	image = tensor.cpu().clone().detach().numpy()
	#clone tensor --> detach it from computations --> transform to numpy
	image = image.squeeze()
	image = image.transpose(1, 2, 0)
	print(image.shape)
	# swap axis from(1,28,28) --> (28,28,1)
	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	#denormalize image
	image = image.clip(0, 1)
	#sets image range from 0 to 1
	return image

mod_gen_path = "models\\model_gan_gen1.pth.tar"
load_gen_mod(torch.load(mod_gen_path))


fixed_noise = torch.randn(1, channels_noise, 1, 1).to(device)



gen_output = gen(fixed_noise).to(device)

plt.imshow(im_convert(gen_output))
plt.show()

