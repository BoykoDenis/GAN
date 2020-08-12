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
import random



class Generator(nn.Module):

	def __init__(self, output_chanels, n_featurs):
		super().__init__()

		self.conv5_1r = nn.ConvTranspose2d(n_featurs*10, n_featurs*5, kernel_size = 5, stride = 1, padding = 0)

		self.upsample_5 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.last_conv5 = nn.ConvTranspose2d(n_featurs*5, n_featurs*5, kernel_size = 5, stride = 1, padding = 2)

		self.upsample_4 = nn.Upsample(scale_factor=2, mode='bilinear') #nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv4 = nn.Conv2d(n_featurs*5, n_featurs*5, kernel_size = 5, stride = 1, padding = 2)

		self.conv4_2r = nn.ConvTranspose2d(n_featurs*5, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv4_1r = nn.ConvTranspose2d(n_featurs*4, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)
		
		self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear') #nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv3 = nn.Conv2d(n_featurs*4, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)

		self.conv3_3r = nn.ConvTranspose2d(n_featurs*4, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_2r = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_1r = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)

		self.upsample_2 = nn.Upsample(scale_factor=(3, 4), mode='bilinear') #nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv2 = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)

		self.conv2_3r = nn.ConvTranspose2d(n_featurs*3, n_featurs*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_2r = nn.Conv2d(n_featurs*2, n_featurs*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_1r = nn.Conv2d(n_featurs*2, n_featurs*2, kernel_size = 5, stride = 1, padding = 2)

		self.upsample_1 = nn.Upsample(scale_factor=(3, 4), mode='bilinear') #nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv1 = nn.Conv2d(n_featurs*2, n_featurs*2, kernel_size = 5, stride = 1, padding = 2)

		self.conv1_3r = nn.ConvTranspose2d(n_featurs*2, n_featurs, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_2r = nn.Conv2d(n_featurs, n_featurs, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_1r = nn.Conv2d(n_featurs, output_chanels, kernel_size = 5, stride = 1, padding = 2)



	def forward(self, x):
		

		x = F.relu(self.conv5_1r(x))

		x = self.upsample_5(x)
		x = F.relu(self.last_conv5(x))

		x = self.upsample_4(x)
		x = F.relu(self.last_conv4(x))
		
		x = F.relu(self.conv4_2r(x))
		x = F.relu(self.conv4_1r(x))
		
		x = self.upsample_3(x)
		x = F.relu(self.last_conv3(x))
		
		x = F.relu(self.conv3_3r(x))
		x = F.relu(self.conv3_2r(x))
		x = F.relu(self.conv3_1r(x))
		
		x = self.upsample_2(x)
		x = F.relu(self.last_conv2(x))
		
		x = F.relu(self.conv2_3r(x))
		x = F.relu(self.conv2_2r(x))
		x = F.relu(self.conv2_1r(x))

		x = self.upsample_1(x)
		x = F.relu(self.last_conv1(x))
		
		x = F.relu(self.conv1_3r(x))
		x = F.relu(self.conv1_2r(x))
		x = F.relu(self.conv1_1r(x))
	
		return x



class Descriminator(nn.Module):

	def __init__(self, input_chanels, n_featurs):
		super().__init__()

		self.conv1_1 = nn.Conv2d(input_chanels, n_featurs, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_2 = nn.Conv2d(n_featurs, n_featurs*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_3 = nn.Conv2d(n_featurs*2, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)

		self.pool1 = nn.MaxPool2d(kernel_size = (3, 4), stride = (3, 4))
		self.conv_last1 = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)

		self.conv2_1 = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_2 = nn.Conv2d(n_featurs*3, n_featurs*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_3 = nn.Conv2d(n_featurs*3, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)

		self.pool2 = nn.MaxPool2d(kernel_size = (3, 4), stride = (3, 4))
		self.conv_last2 = nn.Conv2d(n_featurs*4, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)

		self.conv3_1 = nn.Conv2d(n_featurs*4, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_2 = nn.Conv2d(n_featurs*4, n_featurs*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_3 = nn.Conv2d(n_featurs*4, n_featurs*5, kernel_size = 5, stride = 1, padding = 2)

		self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv_last3 = nn.Conv2d(n_featurs*5, n_featurs*5, kernel_size = 5, stride = 1, padding = 2)

		self.linear1 = nn.Linear(n_featurs*5*20*20, 2)

	def forward(self, x, n_featurs):

		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.relu(self.conv1_3(x))

		x = self.pool1(x)
		x = F.relu(self.conv_last1(x))

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.relu(self.conv2_3(x))

		x = self.pool2(x)
		x = F.relu(self.conv_last2(x))

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))

		x = self.pool3(x)
		x = F.relu(self.conv_last3(x))

		x = x.view(-1, n_featurs*5*20*20)

		x = F.softmax(self.linear1(x))
		#print(self.linear1.weight)

		return x