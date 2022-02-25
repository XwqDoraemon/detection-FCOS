import cv2,os
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from model.config import DefaultConfig
def preprocess_img(image,input_ksize):
    min_side, max_side  = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
if __name__=="__main__":
    model=FCOSDetector(mode="inference",config=DefaultConfig).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./training_dir/model_7.pth",map_location=torch.device('cpu')))
    model=model.eval()
    print("===>success loading model")

    root="/home/myjian/Workespace/PythonProject/Dataset/bdd100k/images/100k/test"
    names=os.listdir(root)

    # img_bgr=cv2.imread(root+name)
    
    start_t = time.time()
    # for i in range(500):
    for i,name in enumerate(names):
        if i%50 == 0:
            print('already tested %d images' % i)
        img_path = os.path.join(root,name)
        img_bgr = cv2.imread(img_path)
        
        img_pad=preprocess_img(img_bgr,[640, 800])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img)
        img1= transforms.Normalize([0.5,0.5,0.5], [1.,1.,1.],inplace=True)(img1)
        img1=img1
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
    end_t=time.time()
    cost_t=1000.*(end_t-start_t)
    print("===>success processing img, the average inference time for each image is %.2f ms"% (cost_t/500.))

