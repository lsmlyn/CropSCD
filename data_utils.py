#coding=utf-8
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(64),
        CenterCrop(64),
        ToTensor()
    ])


def nearest_interpolation(lr_feature_high, fake_image):
    scale = fake_image.size(2)//lr_feature_high.size(2)
    batch_size = lr_feature_high.size(0)
    channels = lr_feature_high.size(1)

    tmp_feature = fake_image  #初始化一个重采样后维度的tensor
    for m in range(batch_size):
        for n in range(channels):
            new_lr_feature_high = lr_feature_high[m][n].unsqueeze(0).unsqueeze(0)
            a = torch.nn.functional.interpolate(new_lr_feature_high, scale_factor = scale,  mode='nearest', align_corners=None)
            tmp_feature[m][n] = a.squeeze()
    return tmp_feature


def getSampleLabel(img_path):
    img_name = img_path.split('\\')[-1]
    return torch.from_numpy(np.array([int(img_name[0] == 'i')], dtype=np.float32))


def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist =[]
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 5).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, args, Time1_dir, Time2_dir, Label_dir):
        super(TestDatasetFromFolder, self).__init__()

        image_sets = [name for name in os.listdir(Time1_dir) for item in args.suffix if
                    os.path.splitext(name)[1] == item]
        self.image1_filenames = [join(Time1_dir, x) for x in image_sets if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in image_sets if is_image_file(x)]
        self.image3_filenames = [join(Label_dir, x) for x in image_sets if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.image3_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 5).squeeze(0)

        image_name =  self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image1_filenames)


class PredictDatasetFromFolder(Dataset):
    def __init__(self,args,  Time1_dir, Time2_dir):
        super(PredictDatasetFromFolder, self).__init__()

        image_sets = [name for name in os.listdir(Time1_dir) for item in args.suffix if
                    os.path.splitext(name)[1] == item]
        self.image1_filenames = [join(Time1_dir, x) for x in image_sets if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in image_sets if is_image_file(x)]


        self.transform = get_transform(convert=True, normalize=False)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))

        image_name = self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name) - 1]

        return image1, image2, image_name

    def __len__(self):
        return len(self.image1_filenames)


class trainImageAug(object):
    def __init__(self, crop = True, augment = True, angle = 30):
        self.crop =crop
        self.augment = augment
        self.angle = angle

    def __call__(self, image1, image2, mask):
        if self.crop:
            w = np.random.randint(0,512)
            h = np.random.randint(0,512)
            box = (w, h, w+512, h+512)
            image1 = image1.crop(box)
            image2 = image2.crop(box)
            mask = mask.crop(box)
        if self.augment:
            prop = np.random.uniform(0, 1)
            if prop < 0.15:
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif prop < 0.3:
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            elif prop < 0.5:
                image1 = image1.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image2 = image2.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                mask = mask.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))

        return image1, image2, mask


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class DA_DatasetFromFolder(Dataset):
    def __init__(self, Image_dir1, Image_dir2, Label_dir, crop=True, augment = True, angle = 30):
        super(DA_DatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug(crop=crop, augment = augment, angle=angle)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index])
        image1, image2, label = self.data_augment(image1, image2, label)
        image1, image2 = self.img_transform(image1), self.img_transform(image2)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 5).squeeze(0)
        return image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, opt, Image_dir1, Image_dir2, Label_dir, Txt_dir):
        super(TrainDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = getDataList(Txt_dir)
        self.n_class = opt.n_class
        self.in_chan = opt.in_chan

        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.lab_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):

        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        # image = image.resize((512, 512), Image.BICUBIC)
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        label = Image.open(self.label_filenames[index])
        if self.n_class == 2:
            label = self.lab_transform(label)
            label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        else:
            label = torch.tensor(np.array(label) / 1.0)
            label = make_one_hot(label.unsqueeze(0).unsqueeze(0).long(), 5).squeeze(0)

        return image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)



class DA_DatasetFromFolder(Dataset):
    def __init__(self, opt, Image_dir1, Image_dir2, Label_dir, Txt_dir, crop=True, augment = True, angle = 30):
        super(DA_DatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = getDataList(Txt_dir)
        self.n_class = opt.n_class
        self.in_chan = opt.in_chan

        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug(crop=crop, augment = augment, angle=angle)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):

        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index])

        image1, image2, label = self.data_augment(image1, image2,label)
        image1 = self.img_transform(image1)
        image2 = self.img_transform(image2)

        if self.n_class ==2:
            label = self.lab_transform(label)
            label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        else:
            label = torch.tensor(np.array(label) / 1.0)
            label = make_one_hot(label.unsqueeze(0).unsqueeze(0).long(), 5).squeeze(0)
        return image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)


class TestDatasetFromFolder(Dataset):
    def __init__(self, opt, Image_dir1, Image_dir2, Label_dir, Txt_dir):
        super(TestDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = getDataList(Txt_dir)
        self.n_class = opt.n_class
        self.in_chan = opt.in_chan

        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.lab_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):

        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        # image = image.resize((512, 512), Image.BICUBIC)
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        label = Image.open(self.label_filenames[index])
        if self.n_class == 2:
            label = self.lab_transform(label)
            label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)
        else:
            label = torch.tensor(np.array(label) / 1.0)
            label = make_one_hot(label.unsqueeze(0).unsqueeze(0).long(), 5).squeeze(0)

        image_name =  self.image_filenames1[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image_filenames1)



