
''' ============================================================== 데이터csv만들기 '''
import numpy as np
from glob import glob
import os
from utils.helper import *
import pandas as pd

plist = glob('../data/cifar10/train/**/*.png',recursive=True)

def parse_(path):
    dir_,filename = os.path.split(path)
    _,class_name = os.path.split(dir_)
    return filename, class_name, path

fnames, cnames, paths = [],[],[]
basket = [fnames,cnames,paths]
for path in plist:
    [ basket[i].append(item) for i,item in enumerate(parse_(path)) ]
    

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

