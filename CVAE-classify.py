# -*- coding: utf-8 -*-
"""
条件变分自编码器（CVAE）
mnist数据集
pytorch深度学习框架
对手写数字图像分类
"""

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os
import numpy as np

os.chdir(os.path.dirname(__file__))

'''模型及损失函数定义'''
'工具函数'
def onehot(x, max_dim):
    'batch_size, -> batch_size, max_dim'
    batch_size = x.shape[0]
    vector = torch.zeros(batch_size, max_dim).to(x.device)
    for i in range(batch_size):
        vector[i,x[i]] = 1
    return vector

'模型结构'
class Encoder(torch.nn.Module):
    #编码器，将input_size维度数据压缩为latent_size维度的mu和sigma
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class Decoder(torch.nn.Module):
    #解码器，将latent_size维度的数据转换为output_size维度的数据
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        x = F.softmax(x) #->bs,output_size #将输出按比例变为和为1的概率值
        return x
    
class CVAE(torch.nn.Module):
    #将编码器解码器组合
    def __init__(self, input_size, output_size, condition_size,
                 latent_size, hidden_size):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size + condition_size, 
                               hidden_size, output_size)
    def forward(self, x,c): #x: bs,input_size|c:bs, condition_size
        #合并输入与条件，输入encoder
        # x = torch.cat((x,c),dim = 1) #bs,input_size + condition_size
        # 压缩，获取mu和sigma
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        #合并采样与条件，输入decoder
        z = torch.cat((z,c),dim = 1) #z: bs,latent_size + condition_size
        # 重构，根据采样数据获取重构数据
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,mu,sigma
    
'损失函数'
#离散分布KL散度，衡量输出10类数字的概率值与输入的onehot值两个分布之间的差异
loss_PROB = torch.nn.KLDivLoss(reduction = 'sum')
#均方误差可作为KL散度替代使用.衡量输出10类数字的概率值与输入的onehot值之间的差异
loss_MSE = torch.nn.MSELoss(reduction = 'sum')

#正态KL散度，衡量正态分布(mu,sigma)与正态分布(0,1)的差异，来源于公式计算
loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)



'''超参数及构造模型'''
'模型参数'
latent_size = 10 #压缩后的特征维度
hidden_size = 20 #encoder和decoder中间层的维度
input_size= output_size = 10 #1讴歌数字类别的维度
condition_size = 28*28 #用28*28的数字图像灰度值作为条件

'训练参数'
epochs = 20 #训练时期
batch_size = 32 #每步训练样本数
learning_rate = 1e-4 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

'构建模型' #如之前训练过，会导入本地已训练的模型文件
modelname = 'vae-label.pth'
model = CVAE(input_size,output_size,condition_size,latent_size,hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
try:
    model.load_state_dict(torch.load(modelname))
    print('[INFO] Load Model complete')
except:
    pass
        
        

''''模型训练、测试、展示''' #如上一步已导入本地模型，可省略本步骤，直接进行 模型推理
'准备mnist数据集' #(数据会下载到py文件所在的data文件夹下)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)#乱序为了测试时每次判断不同的数字图像


loss_history = {'train':[],'eval':[]}
for epoch in range(epochs):   
    '模型训练'
    #每个epoch重置损失 
    train_loss = 0
    #获取数据
    for imgs, lbls in tqdm(train_loader,desc = f'[train]epoch:{epoch}'): #img: (batch_size,1,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0]    
        imgs = imgs.view(bs,condition_size).to(device) #batch_size,input_size(28*28)
        lbls = onehot(lbls.to(device),input_size)

        re_lbls,mu,sigma = model(lbls,imgs)
        #计算损失
        loss_re = loss_PROB(re_lbls, lbls) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
        #反向传播、参数优化，重置梯度
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #记录总损失
        train_loss += loss.item()
    #打印平均损失
    print(f'epoch:{epoch}|TrainLoss: ',train_loss/len(train_loader.dataset))


    '模型测试' #如不需可省略
    model.eval()
    #每个epoch重置损失
    test_loss = 0
    #获取数据
    for imgs, lbls in tqdm(test_loader,desc = f'[eval]epoch:{epoch}'):#img: (batch_size,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0]
        imgs = imgs.view(bs,condition_size).to(device) #batch_size,input_size(28*28)
        lbls = onehot(lbls.to(device),input_size)

        re_lbls,mu,sigma = model(lbls,imgs)
        #计算损失
        loss_re = loss_PROB(re_lbls, lbls) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
        #记录总损失
        test_loss += loss.item()
    #打印平均损失
    print(f'epoch:{epoch}|Test Loss: ',test_loss/len(test_loader.dataset))
    model.train()   
    
    '展示效果' #如不需可省略
    model.eval()
    #图像展示（图像作为输入）
    images = imgs[0].view(28,28) 
    plt.matshow(images.cpu().detach().numpy())
    plt.show()
    #按标准正态分布取样来自造数据
    sample = torch.randn(1,latent_size).to(device)
    images = images.view(1,28*28)
    #拼接数字类别和数字图像
    inputs = torch.cat((sample,images),dim = 1)
    #模型运算
    prob = model.decoder(inputs)[0].cpu().detach().numpy()
    print(prob)#输出模型判断的该数字各个类别的概率值
    print('数字为：', np.argmax(prob)) #概率最大的类别即为判断结果

    model.train()
        
    
    '存储模型'
    torch.save(model.state_dict(), modelname)
        

'''模型推理''' #使用经过 模型训练 的模型或者读取的本地模型进行推理
#对数据集
dataset = datasets.MNIST('/data', train=False, transform=transforms.ToTensor())
#取一组数据
index=0#这是从数据集中取出数据的序号
raw = dataset[index][0].view(1,-1) #raw: bs,28,28->bs,28*28
#图像展示（图像作为输入）
images = imgs[0].view(28,28) 
plt.matshow(images.cpu().detach().numpy())
plt.show()
#按标准正态分布取样来自造数据
sample = torch.randn(1,latent_size).to(device)
images = images.view(1,28*28)
#拼接数字类别和数字图像
inputs = torch.cat((sample,images),dim = 1)
#模型运算
prob = model.decoder(inputs)[0].cpu().detach().numpy()
print(prob)#输出模型判断的该数字各个类别的概率值
print('数字为：', np.argmax(prob)) #概率最大的类别即为判断结果