# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:19:06 2019

@author: Jason Nataprawira
CNN practice using MNIMS (numbers) dataset
"""
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import numpy as np

#Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create a 3x1 matrix in a tensor
#x_data = torch.Tensor([[1.0], [2.0], [3.0]])
#y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__() #use this
		#(input,output,kernelsize)
		#first layer
		#1 channel outputs 6 channels
		self.conv1= nn.Conv2d(1,6,5) #input channel is 3 = RGB, or 1 = grayscale
		#second layer
		self.conv2= nn.Conv2d(6,16,5)
		#third layer
		self.conv3= nn.Conv2d(16,32,5)

		#last before output
		#fully connected 32 channels (depth), 16 width, 16 height
		self.fc1=nn.Linear(32*16*16,10) #compress from 32 to 10 final output

	def forward(self, x):

		#using rectified linear unit
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		#Flatten the tensor
		#print(x.shape)
		x = x.view(-1,32*16*16) 
		x = self.fc1(x)

		return x

#create an instance
#m = Model()

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),batch_size=64, shuffle=True, num_workers=4)


def train(epochs):
	criterion = nn.CrossEntropyLoss()
	#learning rate = how big to shift
	#momentum = smoother
	optimizer = optim.SGD(m.parameters(),lr=0.01, momentum=0.9) 

	for epoch in range(epochs):
		running_loss = 0.0 #initialize loss value
		for i,data in enumerate(train_loader,0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device) #GPU
			optimizer.zero_grad() #reset gradient 

			outputs = m(inputs) #put images to the model 
			loss = criterion(outputs, labels) 
			loss.backward()
			optimizer.step() #optimise the parameters

			#print statistics
			running_loss += loss.item()#loss.data[0]
			#running_loss += loss.item() * inputs.size(0)
			_, preds = torch.max(outputs.data, 1)
			#running_corrects += torch.sum(preds == labels)
			if i % 10 == 9: 
				print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
				running_loss = 0.0
	print('Finished Training...')

def eval():
	#testing the data
	correct = 0
	total = 0

	with torch.no_grad():
		for data in test_loader:
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device) #GPU

			outputs = m(inputs)
			_, predicted = torch.max(outputs.data,1)
			total+= labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy: {:.3f} %'.format(100 * float(correct/total)))

if __name__ == "__main__":
	#training
	m = Model().to(device) #use GPU
	train(1)
	#torch.save(m.state_dict(), ('model.pth')) #save the model

	#testing - still need train and the model too
	#load the model and parameters
	param = torch.load('model.pth', map_location='cuda:0') #map_location default cpu
	m.load_state_dict(param) #load the parameters
	eval()
