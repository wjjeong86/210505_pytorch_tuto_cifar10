
'''
###분류기(Classifier) 학습하기
https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

###custom dataset 클래스 사용.
https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

###model train eval 모드 전환
https://wegonnamakeit.tistory.com/47
'''


MAX_EPOCH = 10


import torch
print(torch.__version__)
print('GPU:',torch.cuda.is_available())

import torchvision
import torchvision.transforms as transforms


''' ============================================================== 데이터csv만들기 '''
import numpy as np
from glob import glob
import os
from utils.helper import *

plist = glob('../data/cifar10/train/**/*.png',recursive=True)

def parse_(path):
    dir_,filename = os.path.split(path)
    _,class_name = os.path.split(dir_)
    return filename, class_name, path

fnames, cnames, paths = [],[],[]
basket = [fnames,cnames,paths]
for path in plist:
    [ basket[i].append(item) for i,item in enumerate(parse_(path)) ]
    
import pandas as pd

basket = [fnames,cnames,paths]
basket = [ list(i) for i in zip(*basket) ]
df = pd.DataFrame(basket, columns=['filename','class_name','path'])
df = df.sort_values(by='path').reset_index(drop=True)

### 클래스 번호 메기기
cnums = []; cname2cnum = {}; count=0
for cname in df['class_name']:
    if cname in cname2cnum.keys():
        pass
    else:
        cname2cnum[cname] = count
        count+=1
    cnums.append(cname2cnum[cname])

### 데이터프레임에 추가
df.insert(2,'class_number',cnums)

### train valid 나누기
fold = np.random.randint(low=0, high=10, size=len(df))
df.insert(3,'fold',fold)

### 저장
df.to_csv('meta_cifar10.csv')
df = pd.read_csv('meta_cifar10.csv',index_col=0)
    
    
### 1 fold만 고르기
sub = df.loc[df['fold']==1]
display(sub)





'''============================================================= data loader '''
'''
본래 dataset은 1개 이미지를 만들고 DataLoader가 배치를 만드는데 사용됨
가장 쉬운 구현을 따라서 구현해봄.

'''
import cv2
from PIL import Image
def imread(P):
    return cv2.cvtColor(cv2.imread(P,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
def imshow(I):
    display(Image.fromarray(np.uint8(np.squeeze(I))))
    
class DatasetCifar10(torch.utils.data.Dataset):
    def __init__(self, meta):
        self.meta = meta
        return
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        tmp_ = self.meta.loc[index]
        path = tmp_['path']
        label = int(tmp_['class_number'])
        image = imread(path) 
        image = np.float32(image)/255.0
        
        return image, label, path
    
meta_tr = df.loc[df['fold']<8].reset_index()
meta_vl = df.loc[df['fold']>=8].reset_index()

ds_tr = DatasetCifar10(meta_tr)
dl_tr = torch.utils.data.DataLoader(dataset = ds_tr, 
                                 batch_size=32,
                                 shuffle=True,
                                )
ds_vl = DatasetCifar10(meta_vl)
dl_vl = torch.utils.data.DataLoader(dataset=ds_vl,
                                   batch_size=32,
                                   shuffle=False,
                                   )


for image, label, path in dl_vl:
    print( image, label, path )    
    break;


    
    
''' =========================================================== model '''

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
from torch import nn
# Define model

def conv33(Cin,Cout):
    return nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=1, padding=1, bias=False)
def bn(Cin):
    return nn.BatchNorm2d(num_features=Cin)
def relu():
    return nn.ReLU()
def pool():
    return nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

def BRC(Cin,Cout):
    return nn.Sequential(
        bn(Cin),
        relu(),
        conv33(Cin,Cout)
    )
    
class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        # 32->16->8->fc
        self.entry_conv_01 = conv33(3,32)
        
        self.body_BRC_01 = BRC(32,32)
        self.body_pool_01 = pool()
        self.body_BRC_02 = BRC(32,64)
        self.body_pool_02 = pool()
        self.body_BRC_03 = BRC(64,128)
        
        self.exit_BRC_01 = BRC(128,128)
        self.exit_BRC_02 = BRC(128,128)
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1) # 어떻게 하든 최종 사이즈만 적으면 되나봄.
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(in_features=128,out_features=10,bias=True)
        

    def forward(self, x):
        
        x = x.permute(0,3,1,2)
        
        z = self.entry_conv_01(x)
        
        z = self.body_BRC_01(z)
        z = self.body_pool_01(z)
        z = self.body_BRC_02(z)
        z = self.body_pool_02(z)
        z = self.body_BRC_03(z)
                
        z = self.exit_BRC_01(z)
        z = self.exit_BRC_02(z)
        
        z = self.GAP(z)
        z = self.flatten(z)
        
        logits = self.fc(z)
        
        return logits

model = MLP3().to(device)
print(model)

pred = model(image.to(device))
pred.shape






'''================================================ loss and opt '''
loss_fn = nn.BCEWithLogitsLoss() # 원핫 인코딩 사용하기 위해서
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




'''================================================== train loop'''
i_epoch = 0 
for i_epoch in range(i_epoch,MAX_EPOCH):
    
    ''' ================ train '''
    model.train()
    for i_batch, (image, label, path) in enumerate(dl_tr):
        ### 데이터 준비
        label_oh = torch.nn.functional.one_hot(label,num_classes=10).float()


        ### 순전파~로스 
        pred = model(image.to(device))
        loss = loss_fn(pred,label_oh.to(device))

        ### 역전파
        optimizer.zero_grad() # 기존 변화도를 지우는(기존 역전파기록을 지우는)
        loss.backward() #이 함수는 backpropagation을 수행하여 x.grad에 변화도를 저장한다.
        
        ### 가중치 업데이트
        optimizer.step()
            
        ### accuracy
        acc = (pred.argmax(1)==label.to(device)).float().mean()

        ### train log display
        if cool_time(key='train_log',cooltime=1.0):
            print(
                f'TR {i_epoch}/{i_batch}   L {loss.item():>7f}   A {acc.item():.3f}'
            )
            
            
    ''' ================ validation '''
    with torch.no_grad():
        model.eval()        
        for i_batch, (image, label, path) in enumerate(dl_vl):
            ### 데이터 준비
            label_oh = torch.nn.functional.one_hot(label,num_classes=10).float()

            ### 순전파~로스
            pred = model(image.to(device))
            loss = loss_fn(pred,label_oh.to(device))
            
            ### accuracy
            acc = (pred.argmax(1)==label.to(device)).float().mean()
            

            ### display
            if cool_time(key='valid_log',cooltime=1.0):
                print(
                    f'VL {i_epoch}/{i_batch}   L {loss.item():>7f}   A {acc.item():.3f}'
                )

    
    
    


