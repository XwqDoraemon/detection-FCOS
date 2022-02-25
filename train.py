import yaml
from model.fcos import FCOSDetector
import torch
# from dataset.VOC_dataset import VOCDataset
from dataset.bdd100k_dataset import BDD100kDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.config import DefaultConfig
from tools.common_tools import *
from tools.progressively_balance import ProgressiveSampler
from torch.nn.utils import clip_grad_value_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_config():
    with open(args.config_file, 'r') as stream:
        opts = yaml.safe_load(stream)
    return opts

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['n_gpu']
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    transform = Transforms()
    config = DefaultConfig
    
    train_dataset = BDD100kDataset(root_dir=opt['data_root_dir'],imgset='10k',scal_mutil=True,mosic_ration = None, 
                    resize_size=[640,960],augment=transform,mean=config.mean,std=config.std)
    
    model = FCOSDetector(mode="training").to(device)
    # model = torch.nn.DataParallel(model)
    

    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']
    LR_INIT= float(opt['LR_INIT'])
    PB = bool(opt['PB'])
    #WARMPUP_STEPS_RATIO = 0.12
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt['n_cpu'], worker_init_fn=np.random.seed(0))
    if PB:
        logger.info("=========PB samper=======")
        sampler_generator = ProgressiveSampler(train_dataset, EPOCHS)
    logger.info("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = 1000

    GLOBAL_STEPS = 1
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(),lr=LR_INIT,momentum=0.9,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-5,last_epoch=-1)

    logger.info(model)
    if opt['chinkpoint']:
        checkpoint_rev = torch.load(opt['chinkpint_path'],map_location=torch.device('cuda'))
        start_epoch = checkpoint_rev['epoch']+1
        optimizer.load_state_dict(checkpoint_rev['optimizer_state_dict'])
        model.load_state_dict(checkpoint_rev['model_state_dict'])
        GLOBAL_STEPS = 1001
    model.train()
    for epoch in range(start_epoch,EPOCHS):
        for epoch_step, data in enumerate(train_loader):
            if PB:
                sampler, p_pb = sampler_generator(epoch)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=False,collate_fn=train_dataset.collate_fn,
                                        num_workers=opt['n_cpu'],
                                        sampler=sampler)
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.to(device)
            batch_boxes = batch_boxes.to(device)
            batch_classes = batch_classes.to(device)

            #lr = lr_func()
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            # if GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
            #     lr = LR_INIT * 0.1
            # for param in optimizer.param_groups:
            #     param['lr'] = lr
            # if GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
            #     lr = LR_INIT * 0.01
            # for param in optimizer.param_groups:
            #     param['lr'] = lr
            start_time = time.time()
            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            if bool(opt['is_clip']):
                clip_grad_value_(model.parameters(), float(opt['clip_val']))
            optimizer.step()
            
            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            if GLOBAL_STEPS%50 == 0:
                logger.info(
                    "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f pb_val:%s" % \
                    (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                    losses[2].mean(), cost_time, optimizer.param_groups[0]['lr'], loss.mean(), str(p_pb)))
            GLOBAL_STEPS += 1
        scheduler.step()
        checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
        torch.save(checkpoint,
            os.path.join(output_dir, "model_{}.pth".format(epoch + 1)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/base.yaml', help='specify config file')
    parser.add_argument('-c','--checkpoint',type=str,default='None',help="checkpoint reserver")
    parser.add_argument('--log',type=str,default='mylog.log',help="checkpoint reserver")
    
    args = parser.parse_args()
    output_dir = '/hy-tmp/training_dir_giou'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #创建log文件
    res_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    logger = make_logger(res_dir,args.log)
    print(args)

    sys.excepthook = handle_exception 
    opt = parse_config()
    main(opt)














