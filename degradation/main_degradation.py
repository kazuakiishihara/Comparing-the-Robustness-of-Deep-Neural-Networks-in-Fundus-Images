# Copyright
# @article{deep_reitna_enhance,
#   title={Modeling and Enhancing Low-Quality Retinal Fundus Images},
#   author={Shen, Ziyi and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
#   journal={IEEE Transactions on Medical Imaging},
#   volume={40},
#   number={3},
#   pages={996--1006},
#   year={2020},
#   publisher={IEEE}
# }

import os
import glob
from multiprocessing.pool import Pool

import numpy as np
from degradation.utils_de import imread, imwrite
from PIL import Image
from degradation.degrad_de import *

import json

sizeX = 224
sizeY = 224

type_list = ['001', '010', '100']
# '111' means: DE_BLUR, DE_SPOT, DE_ILLUMINATION

def process(image_list):  

    for image_path in image_list: 
        name_ext = image_path.split('/')[-1]
        mask_path = os.path.join('./data/mask', name_ext)
        name = name_ext.split('.')[0]
        
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((sizeX, sizeY), Image.BICUBIC)
        mask = np.expand_dims(mask, axis=2)
        mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        
        for t in type_list:
            r_img, r_params = DE_process(img, mask, sizeX, sizeY, t)
            dst_img = os.path.join('./data/de_image', name+'_'+t+'.jpeg')
            imwrite(dst_img, r_img)
            param_dict = {'name':name_ext, 'type':t, 'params':r_params}
            with open(os.path.join('./data/de_js_file', name+'_'+t+'.json'), 'w') as json_file:
                json.dump(param_dict, json_file)

        
if __name__=="__main__":
    
    image_list = glob.glob(os.path.join('./data/image/', '*.jpeg'))
                
    patches = 16
    patch_len = int(len(image_list)/patches)
    filesPatchList = []
    for i in range(patches-1):
        fileList = image_list[i*patch_len:(i+1)*patch_len]
        filesPatchList.append(fileList)
    filesPatchList.append(image_list[(patches-1)*patch_len:])

    # mutiple process
    pool = Pool(patches)
    pool.map(process, filesPatchList)
    pool.close()