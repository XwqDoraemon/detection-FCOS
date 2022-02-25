import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import  Image
import random
import json
from .augment import Transforms,random_perspective
from collections import Counter
def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes

class BDD100kDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "pedestrian",
        "rider",
        "other person",
        "car",
        "bus",
        "truck",
        "train",
        "trailer",
        "other vehicle",
        "motorcycle",
        "bicycle",
        "TL_R",
        "TL_G",
        "TL_Y",
        "TL_NA",
        "traffic sign"
    )
    def __init__(self,root_dir,scal_mutil=False,resize_size=[640,960],imgset='10k',is_train=True, augment=None,mosic_ration = None, mean=[0.5,0.5,0.5], std=[1.,1.,1.]):
        self.root=root_dir
        self.imgset=imgset
        self.train = is_train
        self.scal_mutil = scal_mutil
        if self.train:
            self.labelset = "det_train"
            setform = "train"
        else:
            self.labelset = "det_val"
            setform = "val"
        self._annopath = os.path.join(self.root, "labels","det_20","%s.json")
        self._imgpath = os.path.join(self.root, "images",self.imgset,setform,"%s")

        with open(self._annopath%self.labelset) as f:
            self.img_labels= json.load(f)
        self.images = []
        self.labels = []
        self.label_target = []
        self.name2id=dict(zip(BDD100kDataset.CLASSES_NAME,range(1,len(BDD100kDataset.CLASSES_NAME)+1)))
        self.id2name = {v:k for k,v in self.name2id.items()}
        
        if self.train and self.scal_mutil:
            self.resize_size = [[size,resize_size[1]] for size in range(resize_size[0]//2,resize_size[0]+1,32)]
        else:
            self.resize_size=resize_size
        self.mosic_ration = mosic_ration
        self.mean=mean
        self.std=std
        
        self.augment = augment
        self._get_img_info()
        
        print("INFO=====>bdd100k dataset init finished  ! !")

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self,index):
        # print(self.img_labels[index])
        img_name = self.images[index]
        boxes ,classes = self.labels[index]
        img = Image.open(self._imgpath%img_name)
        
        boxes = np.array(boxes,dtype=np.float32)
        img = np.array(img)
        if self.train:
            if self.scal_mutil:
                resize_size = random.choice(self.resize_size)
            else:
                resize_size = self.resize_size
            if self.mosic_ration is not None:
                if random.random() < self.mosic_ration:
                    img,boxes,classes = self.load_mosaic(index,resize_size)
        else:
            resize_size = self.resize_size
        img,boxes=self.preprocess_img_boxes(img,boxes,resize_size)
    
        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)
        return img,boxes,classes
    
    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
    
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

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes
    def _get_img_info(self):
        for x in self.img_labels:
            # print(self._imgpath%x["name"])
            if not os.path.isfile(self._imgpath%x["name"]):
                continue
            if 'labels' in x:
                self.images.append(x['name'])
                boxes=[]
                classes=[]
                for label in x['labels']:
                    x1 = label['box2d']['x1']
                    y1 = label['box2d']['y1']
                    x2 = label['box2d']['x2']
                    y2 = label['box2d']['y2']
                    box = (x1,y1,x2,y2)
                    boxes.append(box)
                    if label['category'] == 'traffic light':
                        label['category'] = 'TL_'+label['attributes']["trafficLightColor"]
                    classes.append(self.name2id[label['category']])
                self.labels.append((boxes,classes))
        print(len(self.images),len(self.labels))
        assert len(self.images) == len(self.labels)
    def _get_img_num_per_cls(self):
        """
        计算每个类别有多少张样本
        :return:
        """
        img_num_per_cls=[0 for i in range(len(BDD100kDataset.CLASSES_NAME))]
        for label in self.labels:
            boxes,classes = label
            per_class_num = Counter(classes)
            min_num =  min([per_class_num[i] for i in per_class_num.keys()])
            min_classes = [k for k,v in per_class_num.items() if v == min_num]
            min_class = random.choice(min_classes)
            self.label_target.append(min_class-1)
            img_num_per_cls[min_class-1]+=1
        print(img_num_per_cls)
        return img_num_per_cls

    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))

        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))

        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)

        return batch_imgs,batch_boxes,batch_classes

    def load_mosaic(self,index,resize_size):
    #  4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        classes4 = []
        ignore_val = 20
        indices = [index] + random.choices(range(len(self.images)), k=3)  # 3 additional image indices
        s = resize_size
        mosaic_border = [-s[0] // 2, -s[1]// 2]
        yc, xc = (int(random.uniform(-x, 2 * s[i] + x)) for i,x in enumerate(mosaic_border))  # mosaic center x, y

        random.shuffle(indices)
        for i, index in enumerate(indices):
            #classes4 = np.concatenate(classes4,0) Load image
            img_name= self.images[index]
            img = cv2.imread(self._imgpath%img_name)
            h,  w, _  = img.shape
            boxes,classes = self.labels[index]
            
            boxes = np.array(boxes,dtype=np.float32)
            boxes = boxes.copy()
            labels_all = np.concatenate([np.array(classes,dtype=np.float32)[:,None],boxes], axis = 1)
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s[0]* 2, s[1] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
               
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
              
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
              
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
    
            # Labels
            labels_all[:,1] = labels_all[:,1] + padw # top left x
            labels_all[:,2] = labels_all[:,2] + padh # top left y
            labels_all[:,3] = labels_all[:,3] + padw  # bottom right x
            labels_all[:,4] = labels_all[:,4] + padh  # bottom right y
            labels4.append(labels_all)
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        labels4[:,(1,3)] = np.clip(labels4[:,(1,3)],a_min = 0,a_max=s[1]*2)
        labels4[:,(2,4)] = np.clip(labels4[:,(2,4)],a_min = 0,a_max=s[0]*2)
        # 剔除边缘很小的框
        mask = ((labels4[:,3] - 0) > ignore_val ) & ((labels4[:,4] - 0) > ignore_val )\
            & (s[1]*2 - labels4[:,1] > ignore_val) &(s[0]*2 - labels4[:,2] > ignore_val)
        labels4 = labels4[mask]
        # labels4 = labels4[:,1:]
        img4, labels4= random_perspective(img4, labels4, border=mosaic_border)
        return img4,labels4[:,1:],list(labels4[:,0])

if __name__=="__main__":
    # random.seed(2022)
    transform = Transforms()
    dataset = BDD100kDataset(root_dir= "/home/myjian/Workespace/PythonProject/Dataset/bdd100k",imgset='100k',scal_mutil=True,mosic_ration=1,augment=transform)
    print(len(dataset.CLASSES_NAME))
    for i in range(100):
        img,boxes,classes=dataset[i]
        
        img,boxes,classes=img.numpy()*255,boxes.numpy(),classes.numpy()
        print(img.shape)
        img=np.transpose(img,(1,2,0)).astype(np.uint8)

        img = img.copy()
        # print(img)
        # print(classes)
        for i, box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            # print("pt1 is{},{} ".format(pt1,pt2))
            cv2.rectangle(img,pt1,pt2,[0,255,0],1)
            cv2.putText(img, "%s"%(dataset.CLASSES_NAME[int(classes[i]-1)]), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        output_path=os.path.join("/home/myjian/Workespace/PythonProject/Dataset/vision_bdd100k2",str(i)+".jpg")
        # cv2.imshow("test",img)
        cv2.imwrite(output_path, img)
        # if cv2.waitKey(0)==27:
            # break
    # imgs,boxes,classes=dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)








