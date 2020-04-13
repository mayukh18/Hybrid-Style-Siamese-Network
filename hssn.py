import argparse
parser = argparse.ArgumentParser()

# Random seed for the whole run
parser.add_argument('-s', action='store', dest='seed_value', type=int,
                    help='Set the seed value')

# initial learning rate
parser.add_argument('-l', action='store', dest='init_lr', type=float,
                    help='Set the initial learning rate')

# patience for learning rate reduction
parser.add_argument('-p', action='store', type=int, dest='patience',
                    help='patience')

# number of epochs
parser.add_argument('-e', action='store', dest='epochs', type=int,
                    help='number of epochs')

# batch size of the training data generation
parser.add_argument('-b', action='store', dest='batch', type=int,
                    help='batch_size')

# fold id for the cross-validation. considering 5-fold CV
parser.add_argument('-f', action='store', dest='fold_id', type=int,
                    help='fold_id')

# whether to train with the hybrid model. Omitting this would train the
# normal siamese model instead.
parser.add_argument('--hybrid', action='store_true', default=False,
                    dest='hybrid_flag',
                    help='use hybrid model')

results = parser.parse_args()

fstr = 'normal_'
if results.hybrid_flag:
    fstr = 'hybrid_'

# ------------------------------------------------------------------------------------------- #

# logging of results from each epochs.
logfile = open(fstr + str(results.seed_value) + str(results.fold_id) + 'b' + '.txt', 'w')

from utils import printx
printx('seed_value     =', results.seed_value, file=logfile)
printx('hybrid model   =', results.hybrid_flag, file=logfile)
printx('initial LR     =', results.init_lr, file=logfile)
printx('batch size     =', results.batch, file=logfile)
printx('epochs         =', results.epochs, file=logfile)
printx('patience on LR =', results.patience, file=logfile)
printx('fold id =', results.fold_id, file=logfile)

# ------------------------------------------------------------------------------------------- #

SEED_VALUE = results.seed_value
IS_HYBRID = results.hybrid_flag
INITIAL_LR = results.init_lr
BATCH_SIZE = results.batch
EPOCHS = results.epochs
PATIENCE = results.patience
FOLD_ID = results.fold_id

# ------------------------------------------------------------------------------------------- #

import os

os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)

import random

random.seed(SEED_VALUE)

import numpy as np

np.random.seed(SEED_VALUE)

from copy import deepcopy
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision

# results are reproducible with each seed value
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils import MAPScorer, Data_augmentation, jsonf

js = jsonf(fstr + str(SEED_VALUE) + str(FOLD_ID) + ".json")

# -------------------------------------------- Augmentations ------------------------------------------ #

class Data_augmentation:
    def __init__(self):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        return

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def image_augment(self, image):
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        v_flip = random.choice([True, False])
        h_flip = random.choice([True, False])
        # angle = random.choice([0,45,90,135,180,225,270,315,0])
        angle = random.choice([0, -25, 25, -45, 45])
        img_flip = self.flip(image, vflip=False, hflip=h_flip)
        img_rot = self.rotate(img_flip, angle)
        return img_rot

ia = Data_augmentation()

# ------------------------------------------- DATA ------------------------------------------------- #

import pickle

file = open("data/tees.pkl", "rb")
tmp = pickle.load(file)
apparelA_dict = tmp
file.close()

file = open("data/skirts.pkl", "rb")
tmp = pickle.load(file)
apparelB_dict = tmp
file.close()

del tmp

# creating the common image files list

apparelB_list = [key for key in apparelB_dict]
apparelA_list = [key for key in apparelA_dict]

all_list = list(set(apparelA_list).intersection(set(apparelB_list)))
all_list = sorted(all_list)

# train test split for cross validation
from sklearn.model_selection import train_test_split, KFold

ids_train = []
ids_valid = []
skf = KFold(n_splits=5, random_state=SEED_VALUE)
i = 0
for indices in skf.split(X=all_list):
    if i == FOLD_ID:
        ids_train = indices[0]
        ids_valid = indices[1]
        break
    i += 1

base_list = list(np.array(all_list)[ids_train])
test_list = list(np.array(all_list)[ids_valid])

printx("list sizes: all {}, train {}, test {}".format(len(all_list), len(base_list), len(test_list)), file=logfile)


# ----------------------------------------- Data Generators ---------------------------------------- #

class ImDataset(data.Dataset):
    def __init__(self, list_IDs):
        self.data_list = list_IDs
        # print(len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        X = self.data_list[index]
        return X

    def get_exclusive_choice(self, im_name):
        exclusion_base = deepcopy(self.data_list)
        exclusion_base.remove(im_name)
        im_name2 = random.choice(exclusion_base)
        return im_name2


class TripletGenerator(data.Dataset):
    def __init__(self, dataset, seed_value, batch_size=24, init_seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed_value = seed_value
        self.init_seed = init_seed
        self.epoch = 0
        self.total = len(self.dataset)

    def __getitem__(self, index):
        if index >= self.total // self.batch_size + int(self.total % self.batch_size > 0):
            raise IndexError

        random.seed(self.seed_value * (self.init_seed + self.epoch) + index)

        im1 = []
        im2 = []
        im3 = []
        y = []

        # im_names = random.sample(self.data_list, self.batch_size)
        im_names = self.dataset[index * self.batch_size:min(self.total, (index + 1)) * self.batch_size]

        for i in range(len(im_names)):
            im_name = im_names[i]
            im_name2 = self.dataset.get_exclusive_choice(im_name)
            # print("im", im_name, i, index, im_name2)

            bbox_shirt = apparelA_dict[im_name]
            bbox_pant = apparelB_dict[im_name]
            bbox_shirt = ia.image_augment(bbox_shirt)
            bbox_pant = ia.image_augment(bbox_pant)
            bbox_shirt = bbox_shirt / 255.
            bbox_pant = bbox_pant / 255.

            bbox_shirt2 = apparelA_dict[im_name2]
            bbox_pant2 = apparelB_dict[im_name2]
            bbox_shirt2 = ia.image_augment(bbox_shirt2)
            bbox_pant2 = ia.image_augment(bbox_pant2)
            bbox_shirt2 = bbox_shirt2 / 255.
            bbox_pant2 = bbox_pant2 / 255.

            im1.append(bbox_shirt)
            im2.append(bbox_pant)
            im3.append(bbox_pant2)

            im1.append(bbox_pant)
            im2.append(bbox_shirt)
            im3.append(bbox_shirt2)

        final = list(zip(im1, im2, im3))
        random.shuffle(final)
        im1[:], im2[:], im3[:] = zip(*final)

        im1 = np.rollaxis(np.array(im1), 3, 1)
        im2 = np.rollaxis(np.array(im2), 3, 1)
        im3 = np.rollaxis(np.array(im3), 3, 1)

        return [im1, im2, im3]

    def __len__(self):
        return self.total // self.batch_size + int(self.total % self.batch_size > 0)

    def set_epoch_end(self):
        self.epoch += 1


class ValidationGenerator(data.Dataset):
    def __init__(self, dataset, batch_size=50, first=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.first = first
        self.total = len(self.dataset)
        self.max_index = self.total // self.batch_size + int(self.total % self.batch_size > 0)

    def __getitem__(self, index):
        ims = []

        tmp_data = self.dataset[index * self.batch_size:min(self.total, (index + 1)) * self.batch_size]

        for i in range(len(tmp_data)):
            im_name = tmp_data[i]

            if self.first:
                bbox = apparelA_dict[im_name]
                bbox = bbox / 255.
                # bbox = gramRGB(bbox)
            else:
                bbox = apparelB_dict[im_name]
                bbox = bbox / 255.
                # bbox = gramRGB(bbox)

            ims.append(bbox)
        ims = np.rollaxis(np.array(ims), 3, 1)

        return ims

    def __len__(self):
        return self.max_index


# ----------------------------------------- Network ------------------------------------------ #

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        self.dense_layer = nn.Linear(4 * 4 * 512, 256, bias=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self._to_select = ['1', '3', '6', '11']
        del vgg

    def forward(self, x):
        outs = []
        for name, module in self.vgg_features._modules.items():
            x = module(x)
            if name in self._to_select:
                outs.append(x)

        y0 = self.bn0(outs[0])
        y1 = self.bn1(outs[1])
        y2 = self.bn2(outs[2])
        y3 = self.bn3(outs[3])

        out = x.view(-1, 512 * 4 * 4)
        out = self.dense_layer(out)
        return out, [y0, y1, y2, y3]


# ----------------------------------------- Losses ----------------------------------------------- #

class HybridTripletLoss(nn.Module):
    def __init__(self, hybrid=False, margin=1.):
        super(HybridTripletLoss, self).__init__()
        self.margin = margin
        self.style_multiplier = 0.2 * (int(hybrid))

    def forward(self, anchor, positive, negative, tuplep, tuplen):
        p1, p2, p3, p4 = tuplep
        n1, n2, n3, n4 = tuplen

        p1 = p1.view(-1, 64, 128 * 128)
        p2 = p2.view(-1, 64, 128 * 128)
        p3 = p3.view(-1, 128, 64 * 64)
        p4 = p4.view(-1, 256, 32 * 32)

        p1 = torch.bmm(p1, p1.permute(0, 2, 1))
        p2 = torch.bmm(p2, p2.permute(0, 2, 1))
        p3 = torch.bmm(p3, p3.permute(0, 2, 1))
        p4 = torch.bmm(p4, p4.permute(0, 2, 1))

        p1 = p1.view(-1, 64 * 64)
        p2 = p2.view(-1, 64 * 64)
        p3 = p3.view(-1, 128 * 128)
        p4 = p4.view(-1, 256 * 256)

        n1 = n1.view(-1, 64, 128 * 128)
        n2 = n2.view(-1, 64, 128 * 128)
        n3 = n3.view(-1, 128, 64 * 64)
        n4 = n4.view(-1, 256, 32 * 32)

        n1 = torch.bmm(n1, n1.permute(0, 2, 1))
        n2 = torch.bmm(n2, n2.permute(0, 2, 1))
        n3 = torch.bmm(n3, n3.permute(0, 2, 1))
        n4 = torch.bmm(n4, n4.permute(0, 2, 1))

        n1 = n1.view(-1, 64 * 64)
        n2 = n2.view(-1, 64 * 64)
        n3 = n3.view(-1, 128 * 128)
        n4 = n4.view(-1, 256 * 256)

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        distance1 = (p1 - n1).pow(2).sum(1)
        # gramloss1 = F.relu(2 - distance1/(4*64*64*128*128))
        gramloss1 = F.relu(2 - distance1 / (4 * 64 * 64 * 128 * 128))

        distance2 = (p2 - n2).pow(2).sum(1)
        gramloss2 = F.relu(2 - distance2 / (4 * 64 * 64 * 128 * 128))

        distance3 = (p3 - n3).pow(2).sum(1)
        gramloss3 = F.relu(2 - distance3 / (4 * 128 * 128 * 64 * 64))

        # print(p4, n4)

        distance4 = (p4 - n4).pow(2).sum(1)
        gramloss4 = F.relu(2 - distance4 / (4 * 256 * 256 * 32 * 32))

        losses = F.relu(distance_positive - distance_negative + self.margin) + self.style_multiplier * (
                    gramloss1 + gramloss2 + gramloss3 + gramloss4)
        return losses.mean()


# ----------------------------------------------------------------------------------------- #

loader = TripletGenerator(ImDataset(base_list), batch_size=BATCH_SIZE, seed_value=SEED_VALUE, init_seed=0)
valdata = ImDataset(test_list)

htloss = HybridTripletLoss(hybrid=IS_HYBRID)

net = SiameseNet()
net

# ----------------------------------------------------------------------------------------- #

device = 'cuda'

net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=INITIAL_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=PATIENCE, factor=0.5, verbose=True)

# ---------------------------------------- train ------------------------------------------ #

import time

for i in range(EPOCHS):
    cum_loss = 0.
    tm = time.time()
    net = net.train()
    for batch_num, batch in enumerate(loader):
        im1, im2, im3 = batch

        im1 = torch.from_numpy(im1).float().to(device)
        im2 = torch.from_numpy(im2).float().to(device)
        im3 = torch.from_numpy(im3).float().to(device)

        out1, out11 = net(im1)
        out2, out22 = net(im2)
        out3, out33 = net(im3)

        hloss = htloss(out1, out2, out3, out22, out33)

        hloss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cum_loss += hloss
    loader.set_epoch_end()

    printx("epoch {}: time {} loss {}".format(i, time.time() - tm, cum_loss / 143), file=logfile)

    if (i + 1) % 1 != 0:
        continue

    net = net.eval()

    preds_1 = []
    preds_2 = []

    valgen = ValidationGenerator(valdata, first=True)
    for batch_num, batch in enumerate(valgen):
        ims = torch.from_numpy(batch).float().to(device)
        out, _ = net(ims)

        preds_1.append(out.cpu().detach().numpy())

    valgen = ValidationGenerator(valdata, first=False)
    for batch_num, batch in enumerate(valgen):
        ims = torch.from_numpy(batch).float().to(device)
        out, _ = net(ims)

        preds_2.append(out.cpu().detach().numpy())

    preds_1 = np.concatenate(preds_1)
    preds_2 = np.concatenate(preds_2)

    map_1 = MAPScorer(preds_1, preds_2, len(test_list))

    printx("MAP is: ", map_1, file=logfile)
    res = js.read()
    res[i] = map_1
    js.write(res)
    scheduler.step(map_1)

logfile.close()
# ----------------------------------------------------------------------------- #
