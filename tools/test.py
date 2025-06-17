#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import time
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
#import coloredlogs

sys.path.append(os.path.dirname(sys.path[0]))
from utils.initializer import device_initializer, network_initializer, sample_initializer, generate_initializer, seed_initializer
from utils.utils import plot_images, save_images, save_one_image_in_images, check_and_create_dir, save_images_with_name, get_dataset
from utils.checkpoint import load_ckpt
from tools.metrics import get_dice_score, get_iou_score

logger = logging.getLogger(__name__)
#coloredlogs.install(level="INFO")


def test(args):
    """
    Generating
    :param args: Input parameters
    :return: None
    """
    logger.info(msg="Start generation.")
    logger.info(msg=f"Input params: {args}")
    print("GPU :", torch.cuda.get_device_name())
    # Weight path
    weight_path = args.weight_path
    # Run device initializer
    device = device_initializer()
    seed = args.seed
    self_train = args.self_train
    print("current seed", seed)
    seed_initializer(seed_id=seed)
    # Enable conditional generation, sample type, network, image size, number of classes and select activation function
    _, _, network, image_size, _, act = generate_initializer(ckpt_path=weight_path, args=args,
                                                                                      device=device)
    # Saving path
    #result_path = os.path.join(args.result_path, str(time.time()))
    # Check and create result path
    result_path = args.result_path
    check_and_create_dir(result_path)
    # Network
    #Network = network_initializer(network=network, device=device)
    # Initialize the diffusion model
    #diffusion = sample_initializer(sample=sample, image_size=image_size, device=device,sample_step=args.sample_step_ddim)
    # Initialize model
    #Unet = Network(device=device, image_size=image_size, act=act)
    #model = Network().to(device=device)
    #print(self_train)
    if network == "segformer" :
        model = smp.Unet(
            encoder_name="mit_b5",        
            encoder_weights=None,       
            in_channels=3,                      
            classes=1                           
        )
    
    elif network == "unet" :
        model = smp.Unet(
            encoder_name="resnet34",       
            encoder_weights=None,          
            in_channels=3,                 
            classes=1,                      
        )
    model.to(device)
    load_ckpt(ckpt_path=weight_path, model=model, device=device, is_train=False)
    #model.eval() ?

    dataloader_img, dataloader_lbl, dataloader_sam = get_dataset(args=args, dataset_path=args.dataset_path, is_train=False, is_val=False,distributed=False, sam=args.is_sam)
    #sampled_images = diffusion.sample_seg(model=model, img=batch[0].to(device))

    if args.batchnorm :
        model.train()
    else :
        model.eval()
    sum_iou = 0
    sum_dice = 0
    for i, (list_img, list_lbl, list_sam) in tqdm(enumerate(zip(dataloader_img, dataloader_lbl, dataloader_sam))):
        # The images are all resized in dataloader
        images = list_img[0].to(device)
        labels = list_lbl[0][:,0:1,:,:] #1 channel instead of 3 ? (B,1,H,W)
        labels = (labels + 1)/2
        labels = labels.to(device)
        sam = list_sam[0].to(device)
        names = list_img[2]
        
        if self_train :
            #concat = torch.cat([images,sam],dim=1)
            prediction = model(images)
        else :
            prediction = model(images)

        prediction = torch.nn.functional.sigmoid(prediction)
        prediction = prediction > 0.5
        #print(i, get_iou_score(prediction,labels))
        for i in range(prediction.shape[0]) :
            plt.imsave(result_path+"/"+names[i],prediction[i,0,:,:].detach().cpu().numpy())
        #plt.imsave("tmp/"+str(i)+"_lbl.png",labels[0,0,:,:].detach().cpu().numpy())
        sum_iou+= get_iou_score(prediction,labels)
        #print(names, get_iou_score(prediction,labels), get_dice_score(prediction, labels))
        sum_dice+= get_dice_score(prediction, labels)

    print("IoU :", sum_iou/len(dataloader_img))
    print("Dice:", sum_dice/len(dataloader_img))

if __name__ == "__main__":
    # Generating model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Enable conditional generation (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] The conditional settings are consistent with the loaded model training settings.

    # Do we load sam on dataloader ?
    parser.add_argument("--is_sam", type=int, default=1)

    # Batch size for training (required)
    parser.add_argument("--batchnorm", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    parser.add_argument("--conditional", type=bool, default=False)
    parser.add_argument("--num_sample", type=int, default=1)
    # Generation name (required)
    parser.add_argument("--generate_name", type=str, default="df")
    parser.add_argument("--sample_step_ddim", type=int, default=100)

    parser.add_argument("--self_train", type=int, default=0)

    parser.add_argument("--seed", type=int, default=0)
    # Input image size (required)
    # [Warn] Compatible with older versions
    # [Warn] Version <= 1.1.1 need to be equal to model's image size, version > 1.1.1 can set whatever you want
    parser.add_argument("--image_size", type=int, default=64)
    # Number of generation images (required)
    # if class name is `-1` and conditional `is` True, the model would output one image per class.
    parser.add_argument("--num_images", type=int, default=8)
    # Weight path (required)
    parser.add_argument("--weight_path", type=str, default="/your/path/Defect-Diffusion-Model/weight/model.pt")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results/vis")
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm' or 'ddim'.
    # Option: ddpm/ddim
    parser.add_argument("--sample", type=str, default="ddim")
    # Set network
    # Option: unet/cspdarkunet
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's network, version > 1.1.1 can set whatever you want
    parser.add_argument("--network", type=str, default="unet_attention")
    # Set activation function (needed)
    # [Note] The activation function settings are consistent with the loaded model training settings.
    # [Note] If you do not set the same activation function as the model, mosaic phenomenon will occur.
    # Option: gelu/silu/relu/relu6/lrelu
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's act, version > 1.1.1 can set whatever you want
    parser.add_argument("--act", type=str, default="gelu")

    # =====================Enable the conditional generation (if '--conditional' is set to 'True')=====================
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded model training settings.
    # [Warn] Compatible with older versions, version <= 1.1.1
    # [Warn] Version <= 1.1.1 need to be equal to model's num classes, version > 1.1.1 can set whatever you want
    parser.add_argument("--num_classes", type=int, default=10)
    # Class name (required)
    # if class name is `-1`, the model would output one image per class.
    # [Note] The setting range should be [0, num_classes - 1].
    parser.add_argument("--class_name", type=int, default=0)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()
    test(args)
