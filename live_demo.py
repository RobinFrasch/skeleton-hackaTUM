from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from skimage import morphology, img_as_bool, img_as_ubyte
from PIL import Image
import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import pandas as pd

def print_sample_images():
    arm = cv2.imread('assets/showcase_images/arm_raw.png')
    neck = cv2.imread('assets/showcase_images/neck_head_raw.png')
    hand1 = cv2.imread('assets/showcase_images/hand_raw1.png')
    hand2 = cv2.imread('assets/showcase_images/hand_raw2.png')

    plt.subplot(221),plt.imshow(arm),plt.title('Microwave arm Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(neck,cmap = 'gray'),plt.title('Microwave neck/head Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(hand1),plt.title('Microwave hand1 Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(hand2),plt.title('Microwave hand2 Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def preprocessing():
    file_path = "./assets/microwave" #input data

    for index, element in enumerate(os.listdir(file_path)):
        file = os.path.join(file_path, element)
        image = Image.open(file).convert('LA')

        arr = np.array(image)

        # tresholding, blurring
        arr_new = arr[:,:,0]
        blur = cv2.bilateralFilter(arr_new,30,75,75)
        arr_new[blur<70]=0
        arr_new[arr_new!=0]=255

        #out = morphology.medial_axis(arr_new)
        #out = img_as_ubyte(out)
        arr_new = cv2.resize(arr_new, (640, 480), interpolation = cv2.INTER_AREA)
        #out = cv2.resize(out,(640, 480), interpolation = cv2.INTER_AREA)

        target_path = "./assets/preprocessed_out/"
        image_name = "postprocessed_hand_{}.jpg".format(index)
        #"skeletons" were used in an early implementation to detect finger positions using graph-algorithms
        #skeleton_name = "skeleton_hand_{}.jpg".format(index)

        cv2.imwrite(os.path.join(target_path, image_name), arr_new)
        #cv2.imwrite(os.path.join(target_path, skeleton_name), out)

        #plt.subplot(),plt.imshow(image),plt.title('Grayscale Microwave Image'),plt.xticks([]), plt.yticks([])
        #plt.subplot(),plt.imshow(arr_new),plt.title('Preprocessed Image'),plt.xticks([]), plt.yticks([])
        #plt.show()

def dataset_creation():
    file_path_misaligned = "./assets/misaligned" 
    file_path_healthy = "./assets/preprocessed_out" 

    file_misaligned = os.path.join(file_path_misaligned, 'postprocessed_hand_130.jpg')
    file_healthy = os.path.join(file_path_healthy, 'postprocessed_hand_112.jpg')

    img_misaligned = Image.open(file_misaligned)
    img_healthy = Image.open(file_healthy)

    plt.subplot(221),plt.imshow(img_healthy),plt.title('Healthy Palm'),plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(img_misaligned),plt.title('Palm with fractured Finger'),plt.xticks([]), plt.yticks([])
    plt.show()

    return img_misaligned, img_healthy


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        conv_layer1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        
        conv_layer2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        
        self.double_conv = nn.Sequential(
            conv_layer1,# bias was false
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_layer2,# bias was false
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(mean, std)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 3)
        self.down1 = Down(3, 6)
        self.down2 = Down(6, 12)
        self.lin = torch.nn.Linear(12*120*160, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3.flatten(start_dim=1)
        logit = self.lin(x3)
        return torch.nn.functional.sigmoid(logit)

def classification(img_array):
    model = UNet(n_channels=1, n_classes=2)
    parameter_path = './assets/models/classifier_weights'
    print('Path to model exists?')
    print(os.path.exists(parameter_path))
    model.load_state_dict(torch.load(parameter_path))

    img_tensor = torch.tensor(img_array)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if model(img_tensor.float())[0] <= 0.5:
        print("no fracture")
    else:
        print("fracture detected")

#Reading an image
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

class RoadDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

def fraction_prediction():
    # load the model parameters
    device = torch.device('cpu')
    model = BB_model()
    model.load_state_dict(torch.load("./assets/models/model.pth", map_location=device))
    model.eval()

    # resizing test image
    im = read_image('./assets/misaligned/postprocessed_hand_130.jpg')
    im = cv2.resize(im, (int(1.49*300), 300))
    cv2.imwrite('./assets/bb_images/postprocessed_hand_130.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    # test Dataset
    test_ds = RoadDataset(pd.DataFrame([{'path':'./assets/bb_images/postprocessed_hand_130.jpg'}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
    x, y_class, y_bb = test_ds[0]

    xx = torch.FloatTensor(x[None,])
    xx.shape

    # prediction
    out_class, out_bb = model(xx)

    # predicted bounding box
    bb_hat = out_bb.detach().cpu().numpy()
    bb_hat = bb_hat.astype(int)
    show_corner_bb(im, bb_hat[0])
    print(bb_hat[0])