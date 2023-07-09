# -*- coding: utf-8 -*-
"""
CVAE(variant) on mnist
只有decoder接收condition 
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
#交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')
#均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction = 'sum')
#KL散度，衡量正态分布(mu,sigma)与正态分布(0,1)的差异，来源于公式计算
loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)



'''超参数及构造模型'''
'模型参数'
latent_size = 16 #压缩后的特征维度
hidden_size = 128 #encoder和decoder中间层的维度
input_size= output_size = 28*28 #原始图片和生成图片的维度
condition_size = 10 #10个数字，用一个10维向量作为条件

'训练参数'
epochs = 5 #训练时期
batch_size = 32 #每步训练样本数
learning_rate = 3e-3 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

'构建模型' #如之前训练过，会导入本地已训练的模型文件
modelname = 'cvae-variant.pth'
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
    batch_size=batch_size, shuffle=False)


loss_history = {'train':[],'eval':[]}
for epoch in range(epochs):   
    '模型训练'
    #获取数据
    for imgs, lbls in tqdm(train_loader,desc = f'[train]epoch:{epoch}'): 
        #img: (batch_size,1,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0]    
        imgs = imgs.view(bs,input_size).to(device) #batch_size,input_size(28*28)
        lbls = onehot(lbls.to(device),condition_size)
        
        re_imgs,mu,sigma = model(imgs,lbls)
        #计算损失
        loss_re = loss_BCE(re_imgs, imgs) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
        #反向传播、参数优化，重置梯度
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    '模型测试' #如不需可省略
    model.eval()
    #获取数据
    for imgs, lbls in tqdm(test_loader,desc = f'[eval]epoch:{epoch}'):
        #img: (batch_size,28,28)| lbls: (batch_size,)
        bs = imgs.shape[0]
        imgs = imgs.view(bs,input_size).to(device) #batch_size,input_size(28*28)
        lbls = onehot(lbls.to(device),condition_size)
        
        re_imgs,mu,sigma = model(imgs,lbls)
        #计算损失
        loss_re = loss_BCE(re_imgs, imgs) 
        loss_norm = loss_KLD(mu, sigma) 
        loss = loss_re + loss_norm    
  
    model.train()    
    
    '展示效果' #如不需可省略
    model.eval()
    #按标准正态分布取样来自造数据
    sample = torch.randn(1,latent_size).to(device)

    #绘制各个数字的图像
    for i in range(condition_size):
        #用decoder生成新数据
        i_number = i*torch.ones(1).long().to(device)
        condit = onehot(i_number,condition_size)
        inputs = torch.cat((sample,condit),dim = 1)
        gen = model.decoder(inputs)[0].view(28,28)
        plt.matshow(gen.cpu().detach().numpy())
        plt.show()
    model.train()
        
    
    '存储模型'
    torch.save(model.state_dict(), modelname)
        

'''模型推理''' #使用经过 模型训练 的模型或者读取的本地模型进行推理
#按正态分布取样
sample = torch.randn(1,latent_size).to(device)
for i in range(condition_size):
    #用decoder生成新数据
    i_number = i*torch.ones(1).long().to(device)
    condit = onehot(i_number,condition_size)
    inputs = torch.cat((sample,condit),dim = 1)
    gen = model.decoder(inputs)[0].view(28,28)
    plt.matshow(gen.cpu().detach().numpy())
    plt.show()