#!~/.conda/envs/py39/bin/python
import cv2
import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import random
random.seed(2022)
CLASSES_NAME = [
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
        "traffic light",
        "traffic sign"
]

class_numlist = [0 for i in range(len(CLASSES_NAME))]
colors_list = [list(np.random.choice(range(256), size=3)) for i in range(len(CLASSES_NAME))]
name2id = dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
print(colors_list)
if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("-f","--file",help="输入文件")
    parser.add_argument("-i","--imgs",help="输入图片")
    parser.add_argument("-o","--output",help="输出文件")

    args = parser.parse_args()
    with open (args.file,"r") as fr:
        labels = json.load(fr)
    for i, label in tqdm(enumerate(labels)):
        # if i >= 10:
            # break
        img_name = label["name"]
        if "labels" in label:
            img_anns = label["labels"]
            img_path = os.path.join(args.imgs,img_name)
            img = cv2.imread(img_path)
            
            for ann in img_anns:
                x1 = ann['box2d']['x1']
                x2 = ann['box2d']['x2']
                y1 = ann['box2d']['y1']
                y2 = ann['box2d']['y2']
                box=[x1,y1,x2,y2]
                pt1=(int(box[0]),int(box[1]))
                pt2=(int(box[2]),int(box[3]))
                # print(pt1,pt2)
                class_name = ann["category"]
                color_val = colors_list[name2id[class_name]]
                color_val = list([int(x) for x in color_val])  
    
                cv2.rectangle(img,pt1,pt2,color_val,2)
                class_numlist[name2id[class_name]] += 1
                # if class_name not in label_list:
                #     label_list.append(class_name)
                
            output_path = os.path.join(args.output,img_name)
    
            cv2.imwrite(output_path,img)
            # if cv2.waitKey(0)==27:
            #     break
    print(class_numlist)
                
