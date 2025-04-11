from params import par
from model import DeepVO
import cv2
import math
import numpy as np
import time
import torch
import os
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules import loss
from torch import functional as F

# Transform Rotation Matrix to Euler Angle
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

# DataLoader
class KITTI_Data(Dataset):
    def __init__(self, folder, seq_len): 
        # Store image addresses for both image_03 and image_02
        root_train_03 = '/home/krkavinda/Datasets/KITTI_raw/kitti_data/scan/{}/image_03'.format(folder)
        root_train_02 = '/home/krkavinda/Datasets/KITTI_raw/kitti_data/scan/{}/image_02'.format(folder)
        
        imgs_03 = os.listdir(root_train_03)
        imgs_02 = os.listdir(root_train_02)
        self.imgs_03 = [os.path.join(root_train_03, img) for img in imgs_03]
        self.imgs_02 = [os.path.join(root_train_02, img) for img in imgs_02]
        self.imgs_03.sort()
        self.imgs_02.sort()
        
        self.GT = readGT('/home/krkavinda/DeepVO-pytorch/KITTI/pose_GT/{}.txt'.format(folder))
        self.seq_len = seq_len

    def __getitem__(self, index):
        try:
            self.GT[index + self.seq_len]
        except Exception:
            print("Error: Index Out of Range")
        
        # Load paths for both image_03 and image_02
        filenames_03 = [self.imgs_03[index + i] for i in range(self.seq_len + 1)]
        filenames_02 = [self.imgs_02[index + i] for i in range(self.seq_len + 1)]
        
        # Read images and convert to RGB
        images_03 = [np.asarray(cv2.imread(img), dtype=np.float32) for img in filenames_03]
        images_02 = [np.asarray(cv2.imread(img), dtype=np.float32) for img in filenames_02]
        
        images_03 = [img[:, :, (2, 1, 0)] for img in images_03]
        images_02 = [img[:, :, (2, 1, 0)] for img in images_02]
        
        # Transpose to (channels, height, width)
        images_03 = [np.transpose(img, (2, 0, 1)) for img in images_03]
        images_02 = [np.transpose(img, (2, 0, 1)) for img in images_02]
        
        images_03 = [torch.from_numpy(img) for img in images_03]
        images_02 = [torch.from_numpy(img) for img in images_02]
        
        # Stack per 2 images for both modalities
        images_03 = [np.concatenate((images_03[k], images_03[k + 1]), axis=0) for k in range(len(images_03) - 1)]
        images_02 = [np.concatenate((images_02[k], images_02[k + 1]), axis=0) for k in range(len(images_02) - 1)]
        
        # Stack the images for sequences and return both modalities
        return (np.stack(images_03, axis=0), np.stack(images_02, axis=0)), self.GT[index:index + par.seq_len, :]

    def __len__(self):
        return self.GT.shape[0] - 1 - par.seq_len - 1

def readGT(root):
    with open(root, 'r') as posefile:
        GT = []
        for one_line in posefile:
            one_line = one_line.split(' ')
            one_line = [float(pose) for pose in one_line]
            gt = np.append(rotationMatrixToEulerAngles(np.matrix([one_line[0:3], one_line[4:7], one_line[8:11]])), np.array([one_line[3], one_line[7], one_line[11]]))
            GT.append(gt)
    return np.array(GT, dtype=np.float32)

# Custom Loss Function
class DeepvoLoss(loss._Loss):
    def __init__(self, size_average=True, reduce=True):
        super(DeepvoLoss, self).__init__()    

    def forward(self, input, target):
        return F.mse_loss(input[0:3], target[0:3], size_average=self.size_average, reduce=self.reduce) + 100 * F.mse_loss(input[3:6], target[3:6], size_average=self.size_average, reduce=self.reduce)