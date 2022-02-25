import cv2,os
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
# from dataset.VOC_dataset import VOCDataset
from dataset.bdd100k_dataset import BDD100kDataset
import time
from model.config import DefaultConfig

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes
    Returns
    image_paded: input_ksize
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
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

def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                            module.eps, module.momentum,
                            module.affine,
                            module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_name = BDD100kDataset.CLASSES_NAME
    colors_list = [list(np.random.choice(range(256), size=3)) for i in range(len(class_name))]
    name2id = dict(zip(class_name,range(len(class_name))))

    model=FCOSDetector(mode="inference",config=DefaultConfig).to(device)
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load("./training_dir/model_7.pth",map_location=torch.device('cpu')))
    model.load_state_dict(torch.load("./training_dir/model_23.pth", map_location=torch.device('cpu'))["model_state_dict"])
    model=model.eval()
    print("===>success loading model")
    
    root="/home/myjian/Workespace/PythonProject/Dataset/bdd100k/images/100k/test/"
    out_path="/home/myjian/Workespace/PythonProject/detection/FCOS/out_images"
    names=os.listdir(root)
    for name in names:
        img_bgr = cv2.imread(root + name)
        img_pad = preprocess_img(img_bgr, [640,960])
        img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        img1 = transforms.ToTensor()(img)
        img1 = transforms.Normalize([0.5,0.5,0.5], [1.,1.,1.], inplace=True)(img1).to(device)
        start_t=time.time()
        with torch.no_grad():
            out = model(img1.unsqueeze_(dim=0))
        end_t = time.time()
        cost_t = 1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        scores, classes, boxes = out

        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
        
        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))

            color_val = colors_list[name2id[class_name[int(classes[i]-1)]]]
            color_val = list([int(x) for x in color_val]) 
            cv2.rectangle(img_pad, pt1, pt2, color_val,2)
            cv2.putText(img_pad, "%s %.3f"%(class_name[int(classes[i])-1],scores[i]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            print(class_name[int(classes[i])-1], scores[i])
        out_img = os.path.join(out_path,name)
        cv2.imwrite(out_img,img_pad)
        # cv2.imshow('img', img_pad)
        # cv2.waitKey(0)
        




