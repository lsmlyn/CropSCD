# coding=utf-8
import os
import torch.utils.data
from data_utils import TestDatasetFromFolder, calMetric_iou
from models.MeGNet.network import CDNet
import cv2
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--gpu_id', default="1", type=str, help='which gpu to run.')
parser.add_argument('--model_dir', default='epochs/CropSCD//MeGNet.pth', type=str)
parser.add_argument('--memory_dir', default='epochs/CropSCD/MeGNet_memory.pt', type=str)
parser.add_argument('--in_chan', default=3, type=int, help='the number of channels to input the image')
parser.add_argument('--n_class', default=9, type=int, help='the number of types of change')
parser.add_argument('--path_img1', default='dataset/CropSCD/im1', type=str, help='the path of image1 set')
parser.add_argument('--path_img2', default='dataset/CropSCD/im2', type=str, help='the path of image2 set')
parser.add_argument('--path_lab', default='dataset/CropSCD/label', type=str, help='the path of training data label set')
parser.add_argument('--test_txt', default='dataset/CropSCD/test.txt',type=str, help='the path of testing sample name text')
parser.add_argument('--save_dir', default='results/CropSCD/MeGNet/', type=str)

parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

netCD = CDNet(n_class =args.n_class).to(device, dtype=torch.float)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netCD = torch.nn.DataParallel(netCD, device_ids=range(torch.cuda.device_count()))

m_items = torch.load(args.memory_dir)
netCD.load_state_dict(torch.load(args.model_dir))
netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(args, args.path_img1, args.path_img2, args.path_lab, args.test_txt)
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader)

    inter, unin= 0,0

    m_items_test = m_items.clone()

    for hr_img1, lr_img2, label, image_name in test_bar:

        hr_img1 = hr_img1.to(device, dtype=torch.float)
        lr_img2 = lr_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)

        result = netCD(hr_img1, lr_img2, m_items_test,mTrain=False)

        result = torch.argmax(result, 1).unsqueeze(1)
        result = result.cpu().detach().numpy()
        result = np.squeeze(result)

        result = Image.fromarray(result.astype('uint8'))
        result.save(args.save_dir + image_name[0])


