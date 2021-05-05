import GPUtil,os

def set_gpu(gpu_id='AUTO'):
    '''  gpu_id : 'AUTO' : 자동  '1' : 1번 GPU  'CPU' : CPU   '''
    if gpu_id == 'AUTO':       gpu_id = str(GPUtil.getAvailable()[0])
    if gpu_id == 'CPU' :       gpu_id = '-1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return gpu_id



import cv2
def make_brain_mask(images_1,images_2):
        
    black1 = images_1<(10/255)
    black2 = images_2<(10/255)
    white1 = images_1>(250/255)
    white2 = images_2>(250/255)
    
    mask = 1-(black1|black2|white1|white2)
#     kernel = np.ones((7, 7), np.uint8)
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.erode(np.uint8(np.squeeze(mask)), kernel7, iterations=2)
    mask = cv2.dilate(mask, kernel7, iterations=1)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    return mask

def make_ano_image(image, restored_image):
    ''' 
    image : CT
    restored_iamge : Ano 네트워크에서 복원한 이미지
    '''
    diff = np.maximum(image-restored_image-0/255,0)
    return diff

def make_ano_image_refine(image, restored_image):
    ''' 
    image : CT
    restored_iamge : Ano 네트워크에서 복원한 이미지
    '''    
    mask = make_brain_mask(image,restored_image)
    diff = np.maximum(image-restored_image-0/255,0)*mask
#     diff = np.maximum(np.abs(image-restored_image)-0/255,0)*mask  ## 실험중
    diff = np.uint8(diff*255)
    kernel = np.ones((3, 3), np.uint8)
    diff = cv2.erode(np.uint8(np.squeeze(diff)), kernel, iterations=2)
    diff = cv2.dilate(diff, kernel, iterations=2)
    diff = np.clip(np.float32(diff)/255,0,1)
    return diff


import numpy as np
from PIL import Image
def imshow(image):
    from PIL import Image
    try:
        display(Image.fromarray(np.squeeze(np.uint8(image))))
    except:
        print('failed imshow')


    
import numpy as np
# def sigmoid(M):
#     M = 1/(1+np.exp(-M))
#     return M

def sigmoid(x):  
#     return x
    return np.exp(-np.logaddexp(0, -x))


    
def put_text(image,str_):
    import numpy as np
    from PIL import ImageFont, ImageDraw, Image

    image_ = Image.fromarray(np.uint8(image*255))
    font = ImageFont.truetype('fonts/NanumBarunGothic.ttf',20)
    draw = ImageDraw.Draw(image_)
    draw.text((10,10),str_, font=font, fill=(255,255,255,0))
    image = np.float32(image_)/255
    return image



import time
def cool_time(key='default',cooltime=1.0):
    now = time.time()
    if key in cool_time.prevs.keys():
        prev = cool_time.prevs[key]
    else:
        cool_time.prevs[key] = now
        prev = now
    
    if now>(prev+cooltime):
        cool_time.prevs[key] = now
        return True
    else: 
        return False
cool_time.prevs = {}

# if __name__ == '__main__':
#     print(cool_time())
#     time.sleep(0.1)
#     print(cool_time())
#     time.sleep(2.0)
#     print(cool_time())



import os, datetime
def get_log_func( log_path ):
    ''' 로그 남길때 사용하는 함수를 생성하는 함수
        log_path : 로그파일 이름+위치
    '''
    
    def write_log( log_str ):
        ''' 로그 남길때 사용하는 함수, 
            log_str : (string) log string
        '''
        str_ = str(datetime.datetime.now()) + '] ' + log_str + '\n'
        try:
            with open( log_path, 'a+' ) as f:
                f.write( str_ )
        except:
            print('write_log실패')
            pass
        
        print( str_, end='')
        
    return write_log
    
# write_log = get_log_func('log.txt')
# write_log('d')py
    
    
    
    
    
def copy_pyfiles( src_dir, dst_dir, verbose=True ):
    """ 함수가 불려질때 .py 파일을 복사함
        src_dir : (dir path) 복사되길 웡하는 폴더
        dst_dir : (dir path) 복사하길 원하는 폴더, 없으면 만듦.
    """
    import shutil, os, glob
    
    src_dir = os.path.normpath( src_dir )
    dst_dir = os.path.normpath( dst_dir )
    
    # 파일 확인
    list_pyfiles = glob.glob( os.path.join( src_dir, '*.py' ) )
    
    # dst 확인
    if not os.path.exists( dst_dir ):
        os.mkdir( dst_dir )
    
    # 복사
    n_success = 0
    n_failed = 0
    
    for p in list_pyfiles:
        n_failed += 1
        
        _,filename = os.path.split( p )
        src_path = os.path.join( src_dir, filename )
        dst_path = os.path.join( dst_dir, filename )
        
        try :
            shutil.copy( src_path, dst_path )
        except :
            pass
        
        if verbose : print( 'copy {} to {}'.format( src_path, dst_path ) )
            
        n_failed -= 1
        n_success += 1
        
    # 종료구간
    if verbose : print( 'N files {}'.format( len(list_pyfiles) ) )
    if verbose : print( 'n success {}'.format( n_success ) )
    if verbose : print( 'n failed {}'.format( n_failed ) )

if __name__ == '__main__':
    noissy_image = np.random.rand( 100,100 )
    imshow_float32( noissy_image )
    noissy_image = np.random.rand( 100,100,3 )
    imshow_float32( noissy_image )
    
    
import os
def make_savedir(savename):
    '''
    '''
    savepath = os.path.join( 'save', time.strftime( '%y%m%d-%H%M%S' ) + '_' + savename )
    os.makedirs( savepath )
    return savepath


def is_ipython():
    try:
        return True | bool( get_ipython().__class__.__name__ )
    except:
        return False

# print( is_ipython() )