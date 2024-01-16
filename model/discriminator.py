import torch.nn as nn
import torch.nn.functional as F
import torch

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64, pretrain_model=''):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()

		if pretrain_model:
			print('use pretrain model {}'.format(pretrain_model))
			self.init_weight(pretrain_model)


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
	
	def _init_weight(self, pretrain_model):
		state_dict = torch.load(pretrain_model)["state_dict"]
		self_state_dict = self.state_dict()
		for k, v in state_dict.items():
			self_state_dict.update({k: v})
		self.load_state_dict(self_state_dict)