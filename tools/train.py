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
import copy
import logging
#import coloredlogs
import numpy as np
import torch
from scipy.ndimage import label
from skimage.measure import label as label_sk
import random
import segmentation_models_pytorch as smp

from torch import nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(sys.path[0]))
from model.modules.ema import EMA
from utils.initializer import device_initializer, seed_initializer, network_initializer, optimizer_initializer, \
    sample_initializer, lr_initializer, amp_initializer
from utils.utils import plot_images, save_images, get_dataset, setup_logging, save_train_logging, save_images_with_name, DiceLoss
from utils.checkpoint import load_ckpt, save_ckpt
from tools.metrics import get_iou_score, get_dice_score
from model.networks.unet import YNet

logger = logging.getLogger(__name__)
#coloredlogs.install(level="INFO")

def rand_comp_connexe(labels_batch, device) :
    tmp = torch.rand_like(labels_batch)
    assert(tmp.shape==labels_batch.shape)
    for i in range(labels_batch.shape[0]) :
        lbl_i = labels_batch[i].detach().cpu().numpy()
        # Step 1: Get connected components in lbl_i
        labeled_array, num_features = label(lbl_i)

        # Step 2: Randomly delete one connected component
        if num_features > 0:
            component_to_delete = random.randint(1, num_features)  # Randomly select one component
            lbl_modified = np.copy(lbl_i)  # Make a copy of the original image
            lbl_modified[labeled_array == component_to_delete] = 0  # Set the selected component to 0
            tmp[i] = torch.from_numpy(lbl_modified)
        else :
            tmp[i] = torch.from_numpy(lbl_i)
    return tmp.to(device)

def save_batch_comparison(pred, gt, sam=None, save_path='batch_comparison.png'):
    # Ensure predictions and labels are on CPU and converted to numpy
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    
    if sam is not None:
        sam = sam.detach().cpu().numpy()

    batch_size = pred.shape[0]
    
    # Determine the number of columns (2 or 3 depending on sam)
    num_cols = 3 if sam is not None else 2

    # Create subplots with dynamic columns based on the presence of sam
    fig, axes = plt.subplots(nrows=batch_size, ncols=num_cols, figsize=(5 * num_cols, 5 * batch_size))

    for i in range(batch_size):
        # Plot the label (ground truth)
        axes[i, 0].imshow(gt[i, 0], cmap='gray')  # Labels should have shape [batch_size, 1, height, width]
        axes[i, 0].set_title(f'gt {i+1}')
        axes[i, 0].axis('off')

        # Plot the predicted reference mask
        axes[i, 1].imshow(pred[i, 0], cmap='gray')
        axes[i, 1].set_title(f'Prediction {i+1}')
        axes[i, 1].axis('off')

        # Plot the new prediction (if provided)
        if sam is not None:
            axes[i, 2].imshow(sam[i, 0], cmap='gray')
            axes[i, 2].set_title(f'sam {i+1}')
            axes[i, 2].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Batch comparison image saved at {save_path}")

def mix_masks(masks1, masks2):
    batch_size, channels, height, width = masks1.shape
    mixed_masks = masks1.clone()  # Initialize mixed masks

    for i in range(batch_size):
        # Convert masks to numpy for manipulation (use only one channel since all are the same)
        mask1 = masks1[i, 0].cpu().numpy()  # Use the first channel
        mask2 = masks2[i, 0].cpu().numpy()  # Use the first channel

        # Find connected components in the second mask
        labeled_mask2, num_components = label_sk(mask2, return_num=True)

        # Randomly select half of the components
        component_ids = list(range(1, num_components + 1))  # Component IDs start at 1
        random.shuffle(component_ids)
        selected_components = component_ids[:len(component_ids) // 2]

        # Create a mask with the selected components
        selected_mask = np.isin(labeled_mask2, selected_components).astype(np.float32)

        # Mix selected components into the first mask
        mixed_mask = torch.tensor(np.logical_or(mask1, selected_mask).astype(np.float32))

        # Apply the same mixed mask across all channels
        for c in range(channels):
            mixed_masks[i, c] = mixed_mask

    return mixed_masks


def save_masks(masks1, masks2, mixed_masks, idx, file_prefix):
    for c in range(masks1.shape[1]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Mask 1
        axs[0].imshow(masks1[idx, c].cpu(), cmap='gray')
        axs[0].set_title(f'Mask 1 - Channel {c+1}')
        axs[0].axis('off')

        # Plot Mask 2
        axs[1].imshow(masks2[idx, c].cpu(), cmap='gray')
        axs[1].set_title(f'Mask 2 - Channel {c+1}')
        axs[1].axis('off')

        # Plot Mixed Mask
        axs[2].imshow(mixed_masks[idx, c].cpu(), cmap='gray')
        axs[2].set_title(f'Mixed Mask - Channel {c+1}')
        axs[2].axis('off')

        # Save figure as an image
        file_path = f'{file_prefix}_sample{idx}_channel{c+1}.png'
        plt.savefig("test.png", bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid display


def train(rank=None, args=None):
    """
    Training
    :param rank: Device id
    :param args: Input parameters
    :return: None
    """
    logger.info(msg=f"[{rank}]: Input params: {args}")
    seed_initializer(seed_id=args.seed)
    finetune = args.finetune
    sample = args.sample
    network = args.network
    run_name = args.run_name
    image_size = args.image_size
    act = args.act
    init_lr = args.lr
    lr_func = args.lr_func
    num_classes = args.num_classes
    self_train = args.self_train
    with_ynet = args.ynet
    only_sam = args.only_sam
    if only_sam : print("Training with SAM label not refined")
    pt_ckpt = args.pt_ckpt
    unsupervised_extension = args.unsupervised_extension

    # Initialize and save the model identification bit
    distributed = False
    # Run device initializer
    device = device_initializer(device_id=args.use_gpu, is_train=True)
    logger.info(msg=f"[{device}]: Successfully Use normal training.")
    # Whether to enable automatic mixed precision training
    amp = args.amp
    save_model_interval = args.save_model_interval
    start_model_interval = args.start_model_interval
    result_path = args.result_path

    # Create data logging path
    results_logging = setup_logging(save_path=result_path, run_name=run_name)
    results_dir = results_logging[1]
    results_vis_dir = results_logging[2]
    results_tb_dir = results_logging[3]
    
    val_path = os.path.join(results_dir,"tmp")
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        print(f"Directory '{val_path}' created successfully!")

    def memory_stats():
      print(torch.cuda.memory_allocated()/1024**2)
      print(torch.cuda.memory_cached()/1024**2)

    # Dataloader
    print("GPU :", torch.cuda.get_device_name())
    print("source", args.dataset_path)
    print("target", args.dataset_path_trg)
    dataset_path = args.dataset_path
    dataloader_img_src, dataloader_lbl_src, dataloader_sam_src= get_dataset(args=args, dataset_path=dataset_path, is_train=True, distributed=distributed,sam=args.is_sam)
    dataloader_img_trg, dataloader_lbl_trg, dataloader_sam_trg= get_dataset(args=args, dataset_path=args.dataset_path_trg, is_train=True, distributed=distributed,sam=args.is_sam)
    if self_train or with_ynet : 
        dataloader_img_val, dataloader_lbl_val, dataloader_sam_val = get_dataset(args=args, dataset_path=args.dataset_path_trg, is_train=False, is_val=True,distributed=False)
    else : 
        dataloader_img_val, dataloader_lbl_val, dataloader_sam_val = get_dataset(args=args, dataset_path=args.dataset_path, is_train=False, is_val=True,distributed=False, sam=args.is_sam)
    # Network
    #Unet = network_initializer(network=network, device=device)
    #model = Unet(in_channel=6).to(device=device) #if self_train else Unet(in_channel=3).to(device=device)

    if network == "segformer" :
        model = smp.Unet(
            encoder_name="mit_b5",        
            encoder_weights=None,       
            in_channels=3,                      
            classes=1                           
        )
        model.load_state_dict(torch.load('mit_b5.pth'))
        print("segformer weights succefully loaded")
    
    elif network == "unet" :
        model = smp.Unet(
            encoder_name="resnet34",       
            encoder_weights=None,          
            in_channels=3,                 
            classes=1,                      
        )
        model.load_state_dict(torch.load('unet_resnet34_imagenet_weights.pth'))
        print("unet weights succefully loaded")

    # Transfer to device if needed
    model = model.to(device)

    Unet_2 = network_initializer(network="unet", device=device)
    mask_refinement_model = Unet_2(in_channel=3).to(device=device)
    load_ckpt(ckpt_path=pt_ckpt, model=mask_refinement_model, device=device, is_train=False, is_pretrain=True)
    if with_ynet :
        ynet = YNet(n_channels=3,n_classes=2).to(device)
        criterion_ynet = nn.MSELoss()
        Unet = network_initializer(network="unet", device=device)
        model = Unet(in_channel=6).to(device=device) if self_train else Unet(in_channel=3).to(device=device)
    optimizer = optimizer_initializer(model=model, optim='adam', init_lr=init_lr, device=device)
    if finetune :
        print("finetuning refinement network")
        optimizer_refinement = optimizer_initializer(model=mask_refinement_model, optim='adam', init_lr=0.0001, device=device)
    ce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,factor=0.5)

    start_epoch = 0
    scaler = amp_initializer(amp=amp, device=device)

    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    save_train_logging(args, results_dir)
    len_dataloader = len(dataloader_img_src)
    loss_list = []
    loss_list_refinement = []
    best_iou = 0
    best_dice = 0
    best_val = 1000000000
    for epoch in range(start_epoch, args.epochs):
        loss_list_epoch = []
        loss_list_refinement_epoch = []
        current_lr = optimizer.param_groups[0]['lr']
        print("Current Learning Rate:", current_lr)
        model.train()
        if finetune :
            mask_refinement_model.train()
        else :
            mask_refinement_model.eval()
        pbar_img = tqdm(dataloader_img_src)
        for i, (list_img_src, list_lbl, list_img_trg, list_sam_src, list_sam_trg ) in enumerate(zip(pbar_img, dataloader_lbl_src, dataloader_img_trg, dataloader_sam_src, dataloader_sam_trg)):
            # The images are all resized in dataloader
            images_src = list_img_src[0].to(device) #(B,3,H,W)
            labels = list_lbl[0][:,0:1,:,:] #1 channel instead of 3 ? (B,1,H,W)
            labels = (labels + 1)/2 #labels btw 0 and 1
            labels = labels.to(device)
            if self_train or with_ynet:
                images_trg = list_img_trg[0].to(device) #(B,3,H,W)
                sam_src = list_sam_src[0].to(device) #(B,3,H,W)
                sam_trg = list_sam_trg[0].to(device) #(B,3,H,W)
                sam_src = (sam_src + 1)/2
                sam_trg = (sam_trg + 1)/2
            assert list_img_src[2] == list_lbl[2], "The two names are not the same" #To be sure labels are associated to img
            with autocast(enabled=amp):


                if self_train :
                    
                    if not only_sam and not unsupervised_extension :
                        # Mix masks
                        #mixed_masks = mix_masks(sam_src, sam_trg)
                        #save_masks(sam_src, sam_trg, mixed_masks, idx=0, file_prefix='mixed_masks')
                        refined_src = mask_refinement_model(sam_src)
                        refined_trg = (mask_refinement_model(sam_trg)>0.5).to(torch.float32)
                        prediction_src = model(images_src)
                        prediction_trg = model(images_trg)
                        loss_source = ce_loss(prediction_src,labels) + dice_loss(prediction_src,labels)
                        
                        loss_target = ce_loss(prediction_trg,refined_trg[:,0:1,:,:]) #+ dice_loss(prediction_trg,sam_trg[:,0:1,:,:])
                        #loss_refinement = ce_loss(refined_src,labels)
                        #loss_refinement = ce_loss(refined_src,labels) #+ 0.1*ce_loss(refined_mix,labels)
                        loss_refinement = ce_loss(refined_src,labels)
                        loss = loss_source + loss_target

                    if unsupervised_extension and not only_sam : #Warning, finetune should be 0 !!
                        refined_trg = (mask_refinement_model(sam_trg)>0.5).to(torch.float32)
                        prediction_trg = model(images_trg)
                        
                        loss_target = ce_loss(prediction_trg,refined_trg[:,0:1,:,:]) #+ dice_loss(prediction_trg,sam_trg[:,0:1,:,:])
                        #loss_refinement = ce_loss(refined_src,labels)
                        #loss_refinement = ce_loss(refined_src,labels) #+ 0.1*ce_loss(refined_mix,labels)
                        loss_refinement = torch.tensor([0]).to(device)
                        loss = loss_target
                    
                    if unsupervised_extension and only_sam : #Warning, finetune should be 0 !!
                        prediction_trg = model(images_trg)
                        loss_target = ce_loss(prediction_trg,sam_trg[:,0:1,:,:]) #+ dice_loss(prediction_trg,sam_trg[:,0:1,:,:])
                        loss_refinement = torch.tensor([0]).to(device)
                        loss = loss_target

                    
                    elif only_sam and not unsupervised_extension :
                        prediction_src = model(images_src)
                        prediction_trg = model(images_trg)
                        loss_source = ce_loss(prediction_src,labels) + dice_loss(prediction_src,labels)
                        loss_target = ce_loss(prediction_trg,sam_trg[:,0:1,:,:])
                        loss = loss_source + loss_target
                        loss_refinement = torch.tensor([0]).to(device)



            optimizer.zero_grad()
            scaler.scale(loss).backward() #I think should be retain_graph=True ?
            scaler.step(optimizer)
            if finetune :
                optimizer_refinement.zero_grad()
                scaler.scale(loss_refinement).backward()
                scaler.step(optimizer_refinement)
            scaler.update()
            
            # TensorBoard logging
            pbar_img.set_postfix(MSE=loss.item())
            tb_logger.add_scalar(tag=f"[{device}]: MSE", scalar_value=loss.item(),
                                 global_step=epoch * len_dataloader + i)
            loss_list.append((loss.detach().cpu().numpy().item()))
            loss_list_refinement.append(loss_refinement.detach().cpu().numpy().item())
            loss_list_epoch.append((loss.detach().cpu().numpy().item()))
            loss_list_refinement_epoch.append(loss_refinement.detach().cpu().numpy().item())
        # Loss per epoch
        tb_logger.add_scalar(tag=f"[{device}]: Loss", scalar_value=sum(loss_list) / len(loss_list), global_step=epoch)

        # Saving model, set the checkpoint name
        ckpt_model, ckpt_ema_model, ckpt_optimizer = None, None, None
        ckpt_model = model.state_dict()
        ckpt_optimizer = optimizer.state_dict()
        # Enable visualization
        iteration = range(1, len(loss_list) + 1)
        # Plotting the loss
        plt.plot(iteration, loss_list, label='Loss')
        plt.plot(iteration, loss_list_refinement, label='Loss Refinement')
        plt.title('Training Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.ylim((0,1))
        plt.legend()
        plt.savefig(os.path.join(results_vis_dir, f"loss.png"))
        plt.close()
        if epoch%50==0 :
            save_name = "ckpt_epoch_"+str(epoch)
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                    ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                    start_model_interval=start_model_interval, conditional=None, image_size=image_size,
                    sample=sample, network=network, act=act, num_classes=num_classes)
            
        model.eval()
        mask_refinement_model.eval()
        iou = 0
        dice = 0
        loss_val = 0
        loss_refinement_val = 0
        for i, (list_img, list_lbl, list_sam) in tqdm(enumerate(zip(dataloader_img_val, dataloader_lbl_val, dataloader_sam_val))):
            # The images are all resized in dataloader
            images = list_img[0].to(device)
            labels = list_lbl[0][:,0:1,:,:] #1 channel instead of 3 ? (B,1,H,W)
            labels = (labels + 1)/2
            labels = labels.to(device)
            sam = list_sam[0].to(device)
            sam = (sam + 1)/2
            names = list_img[2]

            if self_train :
                #concat = torch.cat([images,sam],dim=1)
                #prediction = model(concat)
                prediction = model(images)
                pred_ref = (mask_refinement_model(sam)>0.5).to(torch.float32)
                loss_val += ce_loss(prediction,sam[:,0:1,:,:]) + ce_loss(prediction,sam[:,0:1,:,:])
                loss_refinement_val += ce_loss(pred_ref,labels)
                if i<3 :
                    save_batch_comparison(pred_ref,labels,sam,os.path.join(results_dir, 'vis','comparison_epoch_'+str(epoch)+"_"+str(i)+'.png'))

            else :
                prediction = model(images)
                loss_ce = ce_loss(prediction,labels)
                loss_dice = dice_loss(prediction, labels)
                loss_val+= loss_ce + loss_dice
            
            prediction = torch.nn.functional.sigmoid(prediction)
            prediction = prediction > 0.5

            iou += get_iou_score(prediction,labels)
            dice += get_dice_score(prediction, labels)
        iou = iou/len(dataloader_img_val)
        dice = dice/len(dataloader_img_val)
        loss_val = loss_val/len(dataloader_img_val)
        loss_val_ref = loss_refinement_val/len(dataloader_img_val)
        print("current loss on val", loss_val_ref)
        print("current iou on val", iou)
        print("best iou", best_iou)
        print("current dice on val", dice)
        print("best dice", best_dice)
        #scheduler.step(loss_val)
        if iou > best_iou :
            best_iou = iou
            save_name = "best_iou"
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                    ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                    start_model_interval=start_model_interval, conditional=None, image_size=image_size,
                    sample=sample, network=network, act=act, num_classes=num_classes)
        if dice > best_dice :
            best_dice = dice
            save_name = "best_dice"
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                    ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                    start_model_interval=start_model_interval, conditional=None, image_size=image_size,
                    sample=sample, network=network, act=act, num_classes=num_classes)
        curr_loss = np.mean(loss_list_epoch)+np.mean(loss_list_refinement_epoch)
        if curr_loss < best_val :
            best_val = curr_loss
            save_name = "best_val"
            print("new best val", curr_loss, "with iou and dice equal to", iou, dice)
            save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                    ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                    start_model_interval=start_model_interval, conditional=None, image_size=image_size,
                    sample=sample, network=network, act=act, num_classes=num_classes)
            



        # Synchronization during distributed training
        if distributed:
            logger.info(msg=f"[{device}]: Synchronization during distributed training.")
            dist.barrier()

    logger.info(msg=f"[{device}]: Finish training.")

    # Clean up the distributed environment
    if distributed:
        dist.destroy_process_group()


def main(args):
    """
    Main function
    :param args: Input parameters
    :return: None
    """
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    else:
        train(args=args)


if __name__ == "__main__":
    # Training model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Set the seed for initialization (required)
    parser.add_argument("--seed", type=int, default=0)
    # Enable conditional training (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] We recommend enabling it to 'True'.
    parser.add_argument("--conditional", type=bool, default=False)
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm' or 'ddim'.
    parser.add_argument("--pt_ckpt", type=str, default="pretrained_cov_shift.pt")
    parser.add_argument("--only_sam", type=int, default=0)
    parser.add_argument("--unsupervised_extension", type=int, default=0)
    # Option: ddpm/ddim
    parser.add_argument("--sample", type=str, default="ddim")
    #train ynet ?
    parser.add_argument("--ynet", type=int, default=0)
    # Do we load sam on dataloader ?
    parser.add_argument("--is_sam", type=int, default=1)
    # Option: unet_attention/cspdarkunet/unet_light
    parser.add_argument("--network", type=str, default="unet_attention")
    # File name for initializing the model (required)
    parser.add_argument("--run_name", type=str, default="df")
    # Total epoch for training (required)
    parser.add_argument("--epochs", type=int, default=3)
    # Batch size for training (required)
    parser.add_argument("--batch_size", type=int, default=2)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=0)
    # Input image size (required)
    parser.add_argument("--image_size", type=int, default=64)
    #Lambda to control the rec loss for the target image
    parser.add_argument("--lambda_rec", type=float, default=1e-3)
    # All images are placed in a single folder, and the path represents the image folder.
    parser.add_argument("--dataset_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    parser.add_argument("--dataset_path_trg", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    # Enable automatic mixed precision training (needed)
    # Effectively reducing GPU memory usage may lead to lower training accuracy and results.
    parser.add_argument("--amp", type=bool, default=False)
    # Set optimizer (needed)
    # Option: adam/adamw
    parser.add_argument("--optim", type=str, default="adamw")
    # Set activation function (needed)
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="gelu")
    # Learning rate (needed)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--self_train", type=int, default=0)
    # Learning rate function (needed)
    # Option: linear/cosine/warmup_cosine
    parser.add_argument("--lr_func", type=str, default="linear")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results")
    # Whether to save weight each training (recommend)
    parser.add_argument("--save_model_interval", type=bool, default=True)
    # Start epoch for saving models (needed)
    # This option saves disk space. If not set, the default is '-1'. If set,
    # it starts saving models from the specified epoch. It needs to be used with '--save_model_interval'
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # Enable visualization of dataset information for model selection based on visualization (recommend)
    parser.add_argument("--vis", type=bool, default=True)
    # Number of visualization images generated (recommend)
    # If not filled, the default is the number of image classes (unconditional) or images.shape[0] (conditional)
    parser.add_argument("--num_vis", type=int, default=10)
    # Resume interrupted training (needed)
    # 1. Set to 'True' to resume interrupted training and check if the parameter 'run_name' is correct.
    # 2. Set the resume interrupted epoch number. (If not, we would select the last)
    # Note: If the epoch number of interruption is outside the condition of '--start_model_interval',
    # it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50,
    # we cannot set any loading epoch points because we did not save the model.
    # We save the 'ckpt_last.pt' file every training, so we need to use the last saved model for interrupted training
    # If you do not know what epoch the checkpoint is, rename this checkpoint is 'ckpt_last'.pt
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=None)
    # Enable use pretrain model (needed)
    parser.add_argument("--pretrain", type=bool, default=False)
    # Pretrain model load path (needed)
    parser.add_argument("--pretrain_path", type=str, default="")
    # Set the use GPU in normal training (required)
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--finetune", type=int, default=0)

    # =================================Enable distributed training (if applicable)=================================
    # Enable distributed training (needed)
    parser.add_argument("--distributed", type=bool, default=False)
    # Set the main GPU (required)
    # Default GPU is '0'
    parser.add_argument("--main_gpu", type=int, default=0)
    # Number of distributed nodes (needed)
    # The value of world size will correspond to the actual number of GPUs or distributed nodes being used
    parser.add_argument("--world_size", type=int, default=2)

    # =====================Enable the conditional training (if '--conditional' is set to 'True')=====================
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded datasets settings.
    parser.add_argument("--num_classes", type=int, default=10)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()

    main(args)
