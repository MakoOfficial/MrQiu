import csv
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import Dataset

import torch.nn.functional as F

import random

from torch.optim.lr_scheduler import StepLR

from albumentations.augmentations.transforms import Lambda, Normalize, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose, Resize

import warnings

import torchvision.transforms as transforms
from utils.func import print

warnings.filterwarnings("ignore")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)


transform_train = Compose([
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    Lambda(image=randomErase)
])

transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])

transform_test = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])


class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        # return (transform_train(image=read_image(f"{self.file_path}/{num}.png"))['image'],
        #         Tensor([row['male']])), row['zscore']
        return (transform_train(image=cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR))['image'],
                # Tensor([row['male']])), Tensor([row['boneage']]).to(torch.int64)
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (transform_val(image=cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR))['image'],
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))
    # for param2 in net.classifer.parameters():
    #     loss += torch.sum(torch.abs(param2))

    return alpha * loss


def L1_penalty_multi(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.module.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def train_fn(net, train_loader, reverse_loader, loss_fn, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss

    net.train()
    iter_length = len(train_loader)
    for batch_idx, (data, reverse_data) in enumerate(zip(train_loader, reverse_loader)):
        optimizer.zero_grad()
        image, gender = data[0]
        reverse_image, reverse_gender = reverse_data[0]
        label = data[1]
        reverse_label = reverse_data[1]

        input_img = torch.cat((image.data, reverse_image.data), dim=0)
        input_gender = torch.cat((gender.data, reverse_gender.data), dim=0)
        input_label = torch.cat((label.data, reverse_label.data), dim=0)

        input_img, input_gender = input_img.type(torch.FloatTensor).cuda(), input_gender.type(torch.FloatTensor).cuda()
        input_label = input_label.type(torch.LongTensor).cuda()

        input_img, input_gender, input_label = shuffle_inputData(input_img, input_gender, input_label)

        batch_size = len(data[1])

        logits = net(input_img, input_gender)

        batch_similarity = cos_similarity(logits) # BxB
        target = get_align_target(input_label, input_gender)
        loss = loss_fn(batch_similarity, target) / 2
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        total_size += batch_size
        if (batch_idx+1) > (iter_length/2):
            break
    return training_loss / total_size


def evaluate_fn(net, val_loader, loss_fn):
    net.eval()
    # net.train()

    feature = torch.zeros((1, 1024), requires_grad=False).type(torch.FloatTensor).cuda()
    total_labels = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    total_genders = torch.zeros((1), requires_grad=False).type(torch.LongTensor).cuda()
    global mae_loss
    global val_total_size
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            label = (data[1] - 1).type(torch.LongTensor).cuda()
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            logits = net(image, gender)
            # label = label.squeeze()
            feature = torch.cat((feature, logits), dim=0)
            label = label.clone().view(-1)
            total_labels = torch.cat((total_labels, label), dim=0)
            gender = gender.clone().view(-1)
            total_genders = torch.cat((total_genders, gender), dim=0)

        feature = feature[1:]
        total_labels = total_labels[1:]
        total_genders = total_genders[1:]
        align_target = get_align_target(total_labels, total_genders, total_labels, total_genders)
        similarity = cos_similarity(feature, feature)
        loss_similarity = loss_fn(similarity, align_target) / 2
        mae_loss = loss_similarity.item()
    return mae_loss


import time
from model import Res50Align, get_My_resnet50


def map_fn(flags):
    model_name = f'Pretrained_align_MrQiu'
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    # gpus = [0, 1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    mymodel = Res50Align(32, *get_My_resnet50(pretrained=True)).cuda()
    #   mymodel.load_state_dict(torch.load('/content/drive/My Drive/BAA/resnet50_pr_2/best_resnet50_pr_2.bin'))
    # mymodel = nn.DataParallel(mymodel.cuda(), device_ids=gpus, output_device=gpus[0])

    train_set, val_set = create_data_loader(train_csvOrder, valid_csv, train_path, valid_path)

    print(train_set.__len__())
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    reverse_set = BAATrainDataset(train_csvReverse, train_path)
    reverse_loader = torch.utils.data.DataLoader(
        reverse_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )


    ## Network, optimizer, and loss function creation

    global best_loss
    best_loss = float('inf')
    # loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_fn = nn.MSELoss(reduction='sum')
    # loss_fn_2 = nn.L1Loss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    ## Trains
    for epoch in range(flags['num_epochs']):
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        global mae_loss
        mae_loss = torch.tensor([0], dtype=torch.float32)
        global val_total_size
        val_total_size = torch.tensor([0], dtype=torch.float32)


        start_time = time.time()
        train_fn(mymodel, train_loader, reverse_loader, loss_fn, optimizer)

        ## Evaluation
        # Sets net to eval and no grad context
        # evaluate_fn(mymodel, val_loader, loss_fn)

        train_loss = training_loss / total_size
        print(
            f'training loss is {train_loss}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))


def get_align_target(labels, gender):
    idx = labels
    labels_mat = torch.index_select(torch.index_select(dis, 0, idx), 1, idx)
    gender = F.one_hot(gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
    gender_mat = torch.matmul(gender, gender.t())
    gender_mat = ((3*gender_mat+1) / 4).data

    return (labels_mat * gender_mat).float().detach()


def cos_similarity(logits):
    logit_nrom = F.normalize(logits)
    similarity = torch.mm(logit_nrom, logit_nrom.t())
    return similarity


def relative_pos_dis():
    dis = torch.zeros((1, 230))
    for i in range(230):
        age_vector = torch.zeros((1, 230))
        if i < 2:
            j = i
            age_vector[0][i + 1] = 1
            age_vector[0][i + 2] = 1
            while j >= 0:
                age_vector[0][j] = 1
                j -= 1
            dis = torch.cat((dis, age_vector), dim=0)
            continue
        if i > 227:
            j = i
            age_vector[0][i - 1] = 1
            age_vector[0][i - 2] = 1
            while j < 230:
                age_vector[0][j] = 1
                j += 1
            dis = torch.cat((dis, age_vector), dim=0)
            continue
        age_vector[0][i-2] = 1
        age_vector[0][i-1] = 1
        age_vector[0][i] = 1
        age_vector[0][i+1] = 1
        age_vector[0][i+2] = 1
        dis = torch.cat((dis, age_vector), dim=0)
    return dis[1:]


def shuffle_inputData(img, gender, label):
    idx = torch.randperm(img.shape[0])
    img, gender, label = img[idx].view(img.size()), gender[idx].view(gender.size()), label[idx].view(label.size())
    return img, gender, label


if __name__ == "__main__":
    save_path = '../../autodl-tmp/Pretrained_50epoch_MrQiu'
    os.makedirs(save_path, exist_ok=True)

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 16
    flags['num_workers'] = 4
    flags['num_epochs'] = 50
    flags['seed'] = 1

    data_dir = '../../autodl-tmp/archive'
    # data_dir = r'E:/code/archive/masked_1K_fold/fold_1'

    train_csvOrder = pd.read_csv(os.path.join(data_dir, "trainOrder.csv"))
    train_csvReverse = pd.read_csv(os.path.join(data_dir, "trainReverse.csv"))
    valid_csv = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    # train_ori_dir = '../../autodl-tmp/ori_4K_fold/'
    # train_ori_dir = '../archive/masked_1K_fold/'

    # delete_diag_mat = delete_diag(flags['batch_size']).cuda()
    dis = relative_pos_dis().detach().cuda()

    print(flags)
    print(f'{save_path} start')
    map_fn(flags)