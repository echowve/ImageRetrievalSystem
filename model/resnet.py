# -*- coding: utf-8 -*-
#深度特征代码，resnet家族，可以有resnet18,34,50,101,152，数据越小，模型越简单
from __future__ import print_function

import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo
import cv2

import numpy as np

means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResidualNet(ResNet):
  def __init__(self, model='resnet50', pretrained=True):
    if model == "resnet18":
        super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    elif model == "resnet34":
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    elif model == "resnet50":
        super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    elif model == "resnet101":
        super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    elif model == "resnet152":
        super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32

    #maxpooling和avgpooling都可以用，这里使用avgpooling
    # max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
    # Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
    # Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
    avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    # fc = self.fc(avg)  # fc.size = N * 1000

    return avg


class ResNetFeat(object):

  def __init__(self, model_name, use_gpu=True):
    
    self.model = ResidualNet(model=model_name)
    self.model.eval()
    self.use_gpu = use_gpu
    if use_gpu:
      if not torch.cuda.is_available():
        print('gpu not found, use cpu instead')
        self.use_gpu = False
    
    if self.use_gpu:
      self.model = self.model.cuda()
    
    print('model {} has been built'.format(model_name))

  def getfeature(self, img):

      #图像归一化，减去训练数据的均值
      img = np.transpose(img, (2, 0, 1)) / 255.
      img[0] -= means[0]  # reduce B's mean
      img[1] -= means[1]  # reduce G's mean
      img[2] -= means[2]  # reduce R's mean
      img = np.expand_dims(img, axis=0)
      if self.use_gpu:
          inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
      else:
          inputs = torch.autograd.Variable(torch.from_numpy(img).float())
      feat =self.model(inputs)

      return feat.data.cpu().numpy().flatten()


if __name__ == "__main__":
  # evaluate database
  FeatureExtractor = ResNetFeat('resnet34')
  feature = FeatureExtractor.getfeature(cv2.imread('E:\\Project\\LogoRetrieval\\logo\\probe\\Armani\\Armani(4).jpg'))
  print(feature.shape)
  
  
