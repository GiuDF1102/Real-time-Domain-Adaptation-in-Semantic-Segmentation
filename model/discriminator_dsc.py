import torch.nn as nn
import torch.nn.functional as F
import torch

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64, pretrain_model=''):
		super(FCDiscriminator, self).__init__()

		self.conv1 = depthwise_separable_conv(num_classes, ndf)
		self.conv2 = depthwise_separable_conv(ndf, ndf*2)
		self.conv3 = depthwise_separable_conv(ndf*2, ndf*4)
		self.conv4 = depthwise_separable_conv(ndf*4, ndf*8)
		self.classifier = depthwise_separable_conv(ndf*8, 1)

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
		
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, groups=nin, kernel_size=4, stride=2, padding=1)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out