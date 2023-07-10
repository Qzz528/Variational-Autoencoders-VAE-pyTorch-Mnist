# -*- coding: utf-8 -*-
"""
变分自编码器（VAE） 使用卷积层
mnist数据集
pytorch深度学习框架
（随机）生成手写数字
"""

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os

os.chdir(os.path.dirname(__file__))

'''模型及损失函数定义'''
'模型结构'
class Encoder(torch.nn.Module):
    #编码器，将input_channel通道的数据变为latent_channel通道
    def __init__(self, input_channel, hidden_channel, latent_channel):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(input_channel, hidden_channel, kernel_size = 13)
        self.mu = nn.Conv2d(hidden_channel, latent_channel, kernel_size = 13)
        self.sigma = nn.Conv2d(hidden_channel, latent_channel, kernel_size = 13)

    def forward(self, x):# x: bs,input_channel,h,w
        x = F.relu(self.conv(x)) #-> bs,hidden_channel,h',w'
        mu = self.mu(x) #-> bs,latent_channel,h",w"
        sigma = self.sigma(x)#-> bs,latent_channel,h",w"
        return mu,sigma

class Decoder(torch.nn.Module):
    #解码器，将latent_channel通道的压缩数据转换为output_channel通道
    def __init__(self, latent_channel, hidden_channel, output_channel):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(latent_channel, hidden_channel, kernel_size = 13)
        self.conv2 = torch.nn.ConvTranspose2d(hidden_channel, output_channel, kernel_size = 13)
    def forward(self, x): # x: bs,latent_channel,h",w"
        x = F.relu(self.conv1(x)) #-> bs,hidden_channel,h',w'
        x = torch.sigmoid(self.conv2(x)) #->bs,input_channel,h,w
        return x
    
class VAE(torch.nn.Module):
    #将编码器解码器组合
    def __init__(self, input_channel, output_channel, latent_channel, hidden_channel):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channel, hidden_channel, latent_channel)
        self.decoder = Decoder(latent_channel, hidden_channel, output_channel)
    def forward(self, x): #x: x: bs,input_channel,h,w
        bs,c = x.shape[:2]
        # 压缩，获取mu和sigma
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_channel,h",w"
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_channel,h",w"
        z = mu + eps*sigma  #z: bs,latent_channel,h",w"
        # 重构，根据采样数据获取重构数据
        re_x = self.decoder(z) # re_x: bs,output_channel,h,w
        return re_x,mu,sigma
    
'损失函数'
#交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')
#均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction = 'sum')
#KL散度，衡量正态分布(mu,sigma)与正态分布(0,1)的差异，来源于公式计算
loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)



'''超参数及构造模型'''
'模型参数'
latent_channel = 4 #提取特征的通道数
hidden_channel = 2 #encoder和decoder中间数据的通道数数
input_channel = output_channel = 1 #原始图片和生成图片的通道数

'训练参数'
epochs = 40 #训练时期
batch_size = 32 #每步训练样本数
learning_rate = 1e-3 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

'构建模型' #如之前训练过，会导入本地已训练的模型文件
modelname = 'vae-conv.pth'
model = VAE(input_channel,output_channel,latent_channel,hidden_channel).to(device)
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
    batch_size=batch_size, shuffle=False)

for epoch in range(epochs):   
    '模型训练'
    #每个epoch重置损失  
    train_loss = 0
    #获取数据
    for imgs, lbls in tqdm(train_loader,desc = f'[train]epoch:{epoch}'): #img: (batch_size,1,28,28)| lbls: (batch_size,)
        imgs = imgs.to(device) #batch_size,1,28,28
        bs = imgs.shape[0]
        
        re_imgs,mu,sigma = model(imgs)
        #计算损失
        loss_re = loss_BCE(re_imgs, imgs) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
        #反向传播、参数优化，重置梯度
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #记录总损失
        train_loss += loss.item()
    #打印该轮平均损失
    print(f'epoch:{epoch}|TrainLoss: ',train_loss/len(train_loader.dataset))

    
    '模型测试' #如不需可省略
    model.eval()
    #每个epoch重置损失
    test_loss = 0
    #获取数据
    for imgs, lbls in tqdm(test_loader,desc = f'[eval]epoch:{epoch}'):#img: (batch_size,28,28)| lbls: (batch_size,)
        imgs = imgs.to(device) #batch_size,1,28,28
        bs = imgs.shape[0]
        
        re_imgs,mu,sigma = model(imgs)
        #计算损失
        loss_re = loss_BCE(re_imgs, imgs) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
        #记录总损失
        test_loss += loss.item()
    #打印该轮平均损失 
    print(f'epoch:{epoch}|Test Loss: ',test_loss/len(test_loader.dataset))
        
    model.train()
    
    
    '展示效果' #如不需可省略
    model.eval()
    #按标准正态分布取样来自造数据
    sample = torch.randn(1,latent_channel,4,4).to(device)
    #宽高均28的图像经过encoder13和13的卷积层，宽高均变为28-13+1-13+1=4
    #因此decoder的输入为bs,latent_channel,4,4
    #用decoder生成新数据
    gen = model.decoder(sample)[0].view(28,28)
    #将测试步骤中的真实数据、重构数据和上述生成的新数据绘图
    concat = torch.cat((imgs[0].view(28, 28),
            re_imgs[0].view( 28, 28), gen), 1)
    plt.matshow(concat.cpu().detach().numpy())
    plt.show()
    model.train()
        
    
    '存储模型'
    torch.save(model.state_dict(), modelname)
        

'''模型推理''' #使用经过 模型训练 的模型或者读取的本地模型进行推理
#按标准正态分布取样来自造数据
sample = torch.randn(1,latent_channel,4,4).to(device)
#宽高均28的图像经过encoder13和13的卷积层，宽高均变为28-13+1-13+1=4
#因此decoder的输入为bs,latent_channel,4,4
#用decoder生成新数据，并转成(28,28)格式
generate = model.decoder(sample)[0].view(28,28)
#展示生成数据
plt.matshow(generate.cpu().detach().numpy())
plt.show()