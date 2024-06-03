# coding=utf-8
import argparse
import os
import pandas as pd
import torch.optim as optim
from torch import nn
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import LoadDatasetFromFolder, TrainDatasetFromFolder, calMetric_iou, DA_DatasetFromFolder
from models.MeGNet.network import CDNet
from loss.losses import cross_entropy
from metrics import update_seg_metrics, get_seg_metrics, eval_metrics
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='Train Change Detection Models')

parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--pretrained', default=0, type=int, help='whether to load the pre-trained model')
parser.add_argument('--pretrain_path', default='epochs/CropSCD/MeGNet/MeGNet.pth', type=str, help='pretrain model path')
parser.add_argument('--pretrain_memory_path', default='epochs/CropSCD/MeGNet/MeGNet_memory.pt', type=str, help='pretrain memory model path')
parser.add_argument('--batchsize', default=8, type=int, help='size of batch')
parser.add_argument('--cropsize', default=512, type=int, help='crop size of the image')
parser.add_argument('--in_chan', default=3, type=int, help='the number of channels to input the image')
parser.add_argument('--n_class', default=9, type=int, help='the number of types of change') #8
parser.add_argument('--mdim', default=64, type=int, help='the dimension of the memory item') #8
parser.add_argument('--n_hybridloss', default=0, type=int, help='coefficient of the hybridloss loss') #8
parser.add_argument('--suffix', default=['.jpg','.png', '.tif'], type=list, help='the suffix of the image')

parser.add_argument('--path_img1', default='dataset/CropSCD/im1', type=str, help='the path of image1 set')
parser.add_argument('--path_img2', default='dataset/CropSCD/im2', type=str, help='the path of image2 set')
parser.add_argument('--path_lab', default='dataset/CropSCD/label', type=str, help='the path of training data label set')
parser.add_argument('--train_txt', default='dataset/CropSCD/train.txt',type=str, help='the path of training sample name text')
parser.add_argument('--val_txt', default='dataset/CropSCD/val.txt', type=str, help='the path of valing sample name text')

parser.add_argument('--save_dir', default='epochs/CropSCD/MeGNet/', type=str, help='the path to save the model')


if __name__ == '__main__':
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    pretrained = opt.pretrained
    batchsize = opt.batchsize
    val_batchsize = 8
    cropsize = opt.cropsize
    mloss = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = DA_DatasetFromFolder(opt, opt.path_img1, opt.path_img2, opt.path_lab, opt.train_txt, crop=False)
    val_set = TrainDatasetFromFolder(opt, opt.path_img1,  opt.path_img2, opt.path_lab, opt.val_txt)

    train_loader = DataLoader(dataset=train_set, num_workers=24, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=24, batch_size=batchsize, shuffle=True)

    netCD = CDNet(n_class =opt.n_class).to(device, dtype=torch.float)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netCD = torch.nn.DataParallel(netCD, device_ids=range(torch.cuda.device_count()))

    if pretrained:
        netCD.load_state_dict(torch.load(opt.pretrain_path))

    # 优化器用什么
    optimizerCD = optim.Adam(netCD.parameters(), lr=0.0001, betas=(0.9, 0.999))
    CDcriterion = cross_entropy().to(device, dtype=torch.float)

    results = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'CD_loss': 0}

        netCD.train()

        for image1, image2, label in train_bar:
            running_results['batch_sizes'] += batchsize

            image1 = Variable(image1)
            image2 = Variable(image2)
            label = Variable(label)

            if torch.cuda.is_available():
                image1 = image1.cuda()
                image2 = image2.cuda()
                label = label.cuda()

            prob, m_items = netCD(image1, image2)

            label = torch.argmax(label, 1)
            CD_loss = CDcriterion(prob, label)

            loss = CD_loss
            netCD.zero_grad()
            loss.backward()
            optimizerCD.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += loss.item() * batchsize

            train_bar.set_description(
                desc='[%d/%d] CD_Loss: %.4f ' % (
                    epoch, NUM_EPOCHS, running_results['CD_loss'] / running_results['batch_sizes'],
                ))

        netCD.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'CD_loss': 0, 'batch_sizes': 0}

            total_inter, total_union = 0, 0
            total_correct, total_label = 0, 0
            total_FPR, total_pred = 0, 0
            num_list = [i for i in range(len(val_loader))]
            pixAcc_list = []
            mIou_list = []
            FPR_list = []

            m_items_test = m_items.clone()

            for image1, image2, label in val_bar:
                valing_results['batch_sizes'] += val_batchsize

                image1 = image1.to(device, dtype=torch.float)
                image2 = image2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)

                prob = netCD(image1, image2, m_items_test, mTrain=False)

                label = torch.argmax(label, 1)

                seg_metrics = eval_metrics(prob, label, num_class=opt.n_class)

                total_inter, total_union, total_correct, total_label, total_FPR, total_pred = update_seg_metrics(
                    total_inter, total_union, total_correct, total_label, total_FPR, total_pred, *seg_metrics)
                pixAcc, mFPR, mIoU, _ = get_seg_metrics(total_correct, total_label, total_inter, total_union, total_FPR,
                                                        total_pred, num_classes=opt.n_class).values()
                val_bar.set_description(
                    'PixeIAcc: {:.4f}, mIoU: {:.4f},mFPR: {:.4f} | '.format(pixAcc, mIoU, mFPR))


        # save model parameters
        if mIoU > mloss:
            mloss = mIoU
            torch.save(netCD.state_dict(), opt.save_dir +  'netCD_epoch_%d.pth' % (epoch))
            torch.save(m_items, opt.save_dir +  'keys_epoch_%d.pt' % (epoch))