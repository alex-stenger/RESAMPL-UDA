import os
import cv2
import numpy as np
from skimage import metrics
import argparse
import torch

def calculate_mean_iou_and_dice(pred_mask, gt_mask, class1_value, class2_value):
    class1_pred_mask = (pred_mask == class1_value)
    class2_pred_mask = (pred_mask == class2_value)
    class1_gt_mask = (gt_mask == class1_value)
    class2_gt_mask = (gt_mask == class2_value)

    intersection_class1 = np.logical_and(class1_pred_mask, class1_gt_mask)
    intersection_class2 = np.logical_and(class2_pred_mask, class2_gt_mask)

    union_class1 = np.logical_or(class1_pred_mask, class1_gt_mask)
    union_class2 = np.logical_or(class2_pred_mask, class2_gt_mask)

    iou_class1 = np.sum(intersection_class1) / (np.sum(union_class1) + 1e-10)
    iou_class2 = np.sum(intersection_class2) / (np.sum(union_class2) + 1e-10)

    mean_iou = (iou_class1 + iou_class2) / 2

    # Dice coefficient calculation
    dice_class1 = 2 * np.sum(intersection_class1) / (np.sum(class1_pred_mask) + np.sum(class1_gt_mask) + 1e-10)
    dice_class2 = 2 * np.sum(intersection_class2) / (np.sum(class2_pred_mask) + np.sum(class2_gt_mask) + 1e-10)

    mean_dice = (dice_class1 + dice_class2) / 2

    return mean_iou, mean_dice

def calculate_iou_and_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    pred_mask = [[1 if element == 255 else element for element in row] for row in pred_mask]
    gt_mask = [[1 if element == 255 else element for element in row] for row in gt_mask]
    iou = np.sum(intersection) / (np.sum(union)+1e-10)
    dice = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-5)

    return iou, dice

def get_dice_score(pd, gt, threshold=0.5):
    """
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: dice coefficient or f1-score
    """

    #pd = (pd > threshold).float()
    intersection = torch.logical_and(pd, gt).sum().item()

    score = (2 * intersection +1e-6) / (torch.sum(pd) + torch.sum(gt) + 1e-6)
    return score.item()


def get_iou_score(pd, gt, threshold=0.5):
    """
    :param threshold:
    :param pd: prediction
    :param gt: ground truth
    :return: iou score or jaccard similarity
    """

    #pd = (pd > threshold).float()
    intersection = torch.logical_and(pd, gt).sum().item()
    union = torch.logical_or(pd, gt).sum().item()

    score = (intersection +1e-6)/ (union + 1e-6)
    return score

def test(pred_path, gt_path, threshold_factor, retour=False):
    global_iou = 0
    global_mean_iou = 0
    global_dice = 0
    total_samples = 0

    for gt_file in os.listdir(gt_path):
        if not gt_file.endswith('.png'):
            continue

        gt_name = gt_file.split('.')[0]
        gt_img = cv2.imread(os.path.join(gt_path, gt_file), cv2.IMREAD_GRAYSCALE)

        pred_masks = []
        for pred_file in os.listdir(pred_path):
            if not pred_file.startswith(gt_name) or not pred_file.endswith('.png'):
                continue

            pred_img = cv2.imread(os.path.join(pred_path, pred_file), cv2.IMREAD_GRAYSCALE)

            # Resize the prediction to match the size of the ground truth
            pred_resized = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

            pred_masks.append(pred_resized)

        if pred_masks:
            # Calculate mean prediction
            pred_mean = np.mean(pred_masks, axis=0).astype(np.uint8)

            # Save mean prediction before thresholding
            mean_save_path = os.path.join(pred_path, f"{gt_name}_mean.png")
            cv2.imwrite(mean_save_path, pred_mean)

            # Threshold the mean prediction
            threshold_value = int(threshold_factor * 255)
            _, pred_thresholded = cv2.threshold(pred_mean, threshold_value, 255, cv2.THRESH_BINARY)

            # Save mean prediction after thresholding
            mean_thresholded_save_path = os.path.join(pred_path, f"{gt_name}_mean_thresholded.png")
            cv2.imwrite(mean_thresholded_save_path, pred_thresholded)

            # Calculate IoU and Dice score for each pair
            iou, dice = calculate_iou_and_dice(pred_thresholded, gt_img)
            mean_iou,_ = calculate_mean_iou_and_dice(pred_thresholded, gt_img, class1_value=255, class2_value=0)

            print(f"GT: {gt_name}, iou: {iou}, dice: {dice}")

            # Accumulate global scores
            if iou > 0 :
                global_iou += iou
                global_mean_iou += mean_iou
                global_dice += dice
                total_samples += 1

    if total_samples > 0:
        # Calculate global scores
        global_iou /= total_samples
        global_dice /= total_samples
        global_mean_iou /= total_samples

        print(f"Global IoU: {global_iou}, Global mean IoU: {global_mean_iou}, Global Dice: {global_dice}")

    if retour :
        return global_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU and Dice scores for predicted masks compared to ground truth masks.")
    parser.add_argument("--pred_path", type=str, help="Path to the directory containing predicted masks.")
    parser.add_argument("--gt_path", type=str, help="Path to the directory containing ground truth masks.")
    parser.add_argument("--threshold_factor", type=float, default=0.5, help="Threshold factor between 0 and 1.")
    args = parser.parse_args()

    test(args.pred_path, args.gt_path, args.threshold_factor)

