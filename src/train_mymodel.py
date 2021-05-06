
'''
###분류기(Classifier) 학습하기
https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

###custom dataset 클래스 사용.
https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

###model train eval 모드 전환
https://wegonnamakeit.tistory.com/47
'''


MAX_EPOCH = 20
VALID_FOLD = [8,9]


import torch, torchvision, torchvision.transforms as transforms
from utils.helper import *
import cv2, pandas as pd
from PIL import Image


print(torch.__version__)
print('GPU:',torch.cuda.is_available())



'''============================================================= data loader '''
'''
본래 dataset은 1개 이미지를 만들고 DataLoader가 배치를 만드는데 사용됨
가장 쉬운 구현을 따라서 구현해봄.

'''
df = pd.read_csv('meta_cifar10.csv',index_col=0)
cond_vl = df['fold'].apply(lambda x: any([x==F for F in VALID_FOLD]))
meta_tr = df.loc[~cond_vl].reset_index()
meta_vl = df.loc[cond_vl].reset_index()



    
class DatasetCifar10(torch.utils.data.Dataset):
    def __init__(self, meta):
        super(DatasetCifar10,self).__init__()
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

ds_tr = DatasetCifar10(meta_tr)
dl_tr = torch.utils.data.DataLoader(dataset = ds_tr,batch_size=32,shuffle=True)
ds_vl = DatasetCifar10(meta_vl)
dl_vl = torch.utils.data.DataLoader(dataset=ds_vl,batch_size=32,shuffle=False)


# for image, label, path in dl_vl:
#     print( image, label, path )    
#     break;


    
    
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
    
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
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

model = mymodel().to(device)
print(model)

# pred = model(image.to(device))
# pred.shape






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

    
    
    
    
'''================================================== save '''

### 가중치만 저장 로드
# torch.save(model.state_dict(),'mmymodelodel.weight')
# model = mymodel()
# model.load_state_dict(torch.load('mymodel.weight'))
# model.eval()


### 모델구조와 가중치 같이 저장
torch.save(model,'mymodel.all')
model = torch.load('mymodel.all')
pred = model(image.to(device)).to('cpu')
max1 = pred.argmax(dim=1)
hit = max1==label
