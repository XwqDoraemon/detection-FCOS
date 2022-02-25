import sys
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_PATH)
# sys.path.insert(1, "/home/myjian/Workespace/PythonProject/detection/FCOS/model")
from .head import ClsCntRegHead
from .fpn_neck import FPN
from .backbone.resnet import resnet50
from .loss import GenTargets,LOSS,coords_fmap2orig,GenDynamicTargets
from .config import DefaultConfig
from .backbone.darknet19 import Darknet19

class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig # 可以看到，如果不输入配置文件，则默认调用上面的配置
        if config.backbone == "resnet50":
            self.backbone = resnet50(pretrained=config.pretrained)
        elif config.backbone == "darknet19":
            self.backbone = Darknet19(pretrained=config.pretrained)

        self.fpn = FPN(config.fpn_out_channels,
                       use_p5=config.use_p5,
                      backbone=config.backbone)

        self.head = ClsCntRegHead(config.fpn_out_channels,
                                  config.class_num,
                                  config.use_GN_head,
                                  config.cnt_on_reg,
                                  config.prior)
        self.config = config

    def train(self, mode=True):
        """
        set module training mode, and frozen bn
        """
        super().train(mode=mode)

    def forward(self, x):
        """
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]

        cls_preds=cls_logits.sigmoid_()
        cnt_preds=cnt_logits.sigmoid_()

        coords =coords.cuda() if torch.cuda.is_available() else coords

        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))#[batch_size,sum(_h*_w)]
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]

        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)

        return scores,classes,boxes

    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)

            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
        #为了使得不同类别的box不重叠，将box的值加一个偏置，这个偏置由一个较大的值*class的id,这样
        #不同的类别的box分布在平面中的不同位置，可以批量计算各个类的nms
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        if mode=="training":
            # self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range)
            self.target_layer = GenDynamicTargets(strides=config.strides,limit_range=config.limit_range)
            self.loss_layer=LOSS()
        elif mode=="inference":
            self.detection_head=DetectHead(config.score_threshold,config.nms_iou_threshold,
                                            config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes=ClipBoxes()

    def forward(self,inputs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode=="training":
            batch_imgs,batch_boxes,batch_classes=inputs
            out=self.fcos_body(batch_imgs)
            targets=self.target_layer([out,batch_boxes,batch_classes])
            losses=self.loss_layer([out,targets])
            return losses
        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net
            '''
            batch_imgs=inputs
            out=self.fcos_body(batch_imgs)
            scores,classes,boxes=self.detection_head(out)
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes

        elif self.mode=="deploy":
            out = self.fcos_body(inputs)
            # Step 1. Concat Cls Output, cls_out shape [1,20,#anchor points]
            cls_out = F.sigmoid(torch.cat([out[0][i].view(1,20,-1) for i in range(len(out[0]))], -1))
            # Step 2. Concat Cnt Output, cnt_out shape [1,1,#anchor points]
            cnt_out = F.sigmoid(torch.cat([out[1][i].view(1,1,-1) for i in range(len(out[1]))], -1))
            # Step 3. Concat Reg Output, reg_out shape [1,4,#anchor points]
            reg_out = torch.cat([out[2][i].view(1,4,-1) for i in range(len(out[2]))], -1)
            return cls_out,cnt_out,reg_out

if __name__ == "__main__":
   x =  torch.randn(1,3,320,480)
   model = FCOS()
   out = model(x)
   for i in range(3):
       print(out[i].shape)

