from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os

def update_seg_metrics(total_inter,total_union,total_correct,total_label,total_FPR,total_pred,correct, labeled, inter, union,FPR,pred):
    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union
    total_FPR += FPR
    total_pred += pred
    return total_inter,total_union,total_correct,total_label,total_FPR,total_pred

def get_seg_metrics(total_correct,total_label,total_inter,total_union,total_FPR,total_pred,num_classes):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    FPR = 1.0 * total_FPR / (np.spacing(1) + total_pred)
    mFPR = FPR.mean()
    mIoU = IoU.mean()
    return {
        "Pixel_Accuracy": np.round(pixAcc, 5),
        "Mean_FPR":np.round(mFPR,5),
        "Mean_IoU": np.round(mIoU, 5),
        "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 5)))
    }


def batch_pix_accuracy(predict, target, labeled):
    # 计算标签的总和，是一个batch中的所有标签的总数
    # 注意  python中默认的T为1 F为0  调用sum就是统计正确的像素点的个数
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()   # 将一个batch中预测正确的，且在标签范围内的像素点的值统计出来
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()   # 这里得到的就是所有标签的像素点的总数，和所有预测正确的像素点的总数

#   计算相同类别的IoU
def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()   # 返回预测中在指定范围内的像素点，保证分类类别在指定范围
    intersection = predict * (predict == target).long()   # 过滤掉预测中不正确的像素值
    # intersection.size() (4 224 224)  一个batch中只有正确的像素值才在intersection中，不正确的为0
    # torch.histc 统计图片中的从0-bins出现的次数，返回一个列表
    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)   # area_inter 会得到batch中每个类别 对应像素点(分类正确的)出现了多少次
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)         # area_pred 将batch中预测的所有像素点(不管正不正确) 在每个类别的次数统计出来
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)           # area_lab 将batch中每个类别实际有多少像素点统计出来
    area_union = area_pred + area_lab - area_inter                                         # 预测与标签相交的部分 每个类别对应像素点的数量
    area_FPR = area_pred - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy(),area_FPR.cpu().numpy(),area_pred.cpu().numpy()

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1)    #  按通道维度取最大，拿到每个像素点分类的类别
    predict = predict + 1                     # 每个都加1避免从0开始
    target = target + 1

    labeled = (target > 0) * (target <= num_class)   # 得到一个矩阵，其中，为true的是1，为false的是0   标签中同时满足大于0 小于num_classes 的地方为T,其余地方为F  构成了一个蒙版
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)   # 计算一个batch中预测正确像素点的个数和所有像素点的总数
    inter, union, area_FPR,area_pred = batch_intersection_union(predict, target, num_class, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5),np.round(area_FPR,5),np.round(area_pred,5)]