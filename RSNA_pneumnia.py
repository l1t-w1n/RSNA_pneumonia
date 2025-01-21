import os
import time

import skimage
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.ndimage import map_coordinates, gaussian_filter  # Updated imports to avoid deprecation warnings
import PIL

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision as tv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

import shutil

import pydicom

from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

gpu_available = True
device = torch.device('cuda' if torch.cuda.is_available() and gpu_available else 'cpu')

original_image_shape = 1024

datapath_prep = "./RSNA_pneumonia/"

# read train dataset
df_train = pd.read_csv(datapath_prep + 'train.csv')
# read test dataset
df_test = pd.read_csv(datapath_prep + 'test.csv')
print(df_train.head(3))

validation_frac = 0.10

df_train = df_train.sample(frac=1, random_state=42) 
pIds = [pId for pId in df_train['patientId'].unique()]

pIds_valid = pIds[: int(round(validation_frac * len(pIds)))]
pIds_train = pIds[int(round(validation_frac * len(pIds))):]

print('{} patient IDs shuffled and {}% of them used in validation set.'.format(len(pIds), validation_frac * 100))
print('{} images went into train set and {} images went into validation set.'.format(len(pIds_train), len(pIds_valid)))

# get test set patient IDs
pIds_test = df_test['patientId'].unique()
print('{} patient IDs in test set.'.format(len(pIds_test)))


def get_boxes_per_patient(df, pId):
    boxes = df.loc[df['patientId'] == pId][['x', 'y', 'width', 'height']].astype('int').values.tolist()
    return boxes


pId_boxes_dict = {}
for pId in df_train.loc[(df_train['Target'] == 1)]['patientId'].unique().tolist():
    pId_boxes_dict[pId] = get_boxes_per_patient(df_train, pId)
print('{} ({:.1f}%) images have target boxes.'.format(len(pId_boxes_dict), 100 * (len(pId_boxes_dict) / len(pIds))))


def imgMinMaxScaler(img, scale_range):
    warnings.filterwarnings("ignore")
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    # round at closest integer and transform to integer
    img_scaled = np.rint(img_scaled).astype('uint8')
    return img_scaled


def elastic_transform(image, alpha, sigma, random_state=None):
    assert len(image.shape) == 2, 'Image must have 2 dimensions.'
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    image_warped = map_coordinates(image, indices, order=1).reshape(shape)
    return image_warped


class PneumoniaDataset(torchDataset):
    """
        Pneumonia dataset that contains radiograph lung images as .dcm.
        Each patient has one image named patientId.dcm.
    """

    def __init__(self, root, subset, pIds, predict, boxes, rescale_factor=1, transform=None, rotation_angle=0,
                 warping=False):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, if true returns images and target labels, otherwise returns only images
        :param boxes: a {patientId : list of boxes} dictionary (ex: {'pId': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        :param rescale_factor: image rescale factor in network (image shape is supposed to be square)
        :param transform: transformation applied to the images and their target masks
        :param rotation_angle: float, defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        :param warping: boolean, whether applying augmentation warping to image
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        if self.subset not in ['train', 'test']:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of: \'train\' or \'test\'')
        self.pIds = pIds
        self.predict = predict
        self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.data_path = self.root + 'stage_2_' + self.subset + '_images/'

    def __getitem__(self, index):
        pId = self.pIds[index]
        img = pydicom.dcmread(os.path.join(self.data_path, pId + '.dcm')).pixel_array
        if img.shape[0] != img.shape[1]:
            raise RuntimeError('Image shape {} should be square.'.format(img.shape))
        original_image_shape = img.shape[0]
        image_shape = original_image_shape // self.rescale_factor
        if image_shape != int(image_shape):
            raise RuntimeError('Network image shape should be an integer.'.format(image_shape))
        image_shape = int(image_shape)
        img = resize(img, (image_shape, image_shape), mode='reflect')
        img = imgMinMaxScaler(img, (0, 255))
        if self.warping:
            img = elastic_transform(img, image_shape * 2., image_shape * 0.1)
        img = np.expand_dims(img, -1)
        img = to_pil_image(img)

        if self.rotation_angle > 0:
            angle = self.rotation_angle * (2 * np.random.random_sample() - 1)
            img = img.rotate(angle, resample=PIL.Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        if not self.predict:
            target = np.zeros((image_shape, image_shape))
            if pId in self.boxes:
                for box in self.boxes[pId]:
                    x, y, w, h = box
                    x = int(round(x / self.rescale_factor))
                    y = int(round(y / self.rescale_factor))
                    w = int(round(w / self.rescale_factor))
                    h = int(round(h / self.rescale_factor))
                    target[y:y + h, x:x + w] = 255
                    target[target > 255] = 255
            target = np.expand_dims(target, -1)
            target = target.astype('uint8')
            target = to_pil_image(target)

            if self.rotation_angle > 0:
                target = target.rotate(angle, resample=PIL.Image.BILINEAR)

            if self.transform is not None:
                target = self.transform(target)

            return img, target, pId
        else:
            return img, pId

    def __len__(self):
        return len(self.pIds)


# manual model parameters
rescale_factor = 4  # resize factor to reduce image size (new_image_shape = original_image_shape / rescale_factor)
batch_size = 10  # Adjust batch size as needed
num_workers = 20  # Number of CPU cores to use for data loading

# recalculate minimum box area
min_box_area = 10000
min_box_area = int(round(min_box_area / float(rescale_factor ** 2)))

# TBD add normalization of images into transforms
# define transformation
transform = tv.transforms.Compose([tv.transforms.ToTensor()])

# create datasets
dataset_train = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_train, predict=False,
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=3, warping=True)

dataset_valid = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_valid, predict=False,
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=0, warping=False)

dataset_test = PneumoniaDataset(root=datapath_prep, subset='test', pIds=pIds_test, predict=True,
                                boxes=None, rescale_factor=rescale_factor, transform=transform,
                                rotation_angle=0, warping=False)

# define the dataloaders with the previous dataset
loader_train = DataLoader(dataset=dataset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

loader_valid = DataLoader(dataset=dataset_valid,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

loader_test = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)

# Check if train images have been properly loaded
print('{} images in train set, {} images in validation set, and {} images in test set.'.format(len(dataset_train),
                                                                                               len(dataset_valid),
                                                                                               len(dataset_test)))
img_batch, target_batch, pId_batch = next(iter(loader_train))
print('Tensor batch size:', img_batch.size())

# Define the UNet model
class conv_block(nn.Module):
    """
    Define the [convolution - batch normalization - activation] block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))


# Define the nn transposed convolutional block
class conv_t_block(nn.Module):
    """
    Define the [convolution_transpose - batch normalization - activation] block
    """

    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_t_block, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1,
                                         bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x, output_size):
        return self.activ(self.bn(self.conv_t(x, output_size=output_size)))


class PneumoniaUNET(nn.Module):

    def __init__(self):
        super(PneumoniaUNET, self).__init__()

        self.down_1 = nn.Sequential(conv_block(in_channels=1, out_channels=64),
                                    conv_block(in_channels=64, out_channels=64))
        self.down_2 = nn.Sequential(conv_block(in_channels=64, out_channels=128),
                                    conv_block(in_channels=128, out_channels=128))
        self.down_3 = nn.Sequential(conv_block(in_channels=128, out_channels=256),
                                    conv_block(in_channels=256, out_channels=256))
        self.down_4 = nn.Sequential(conv_block(in_channels=256, out_channels=512),
                                    conv_block(in_channels=512, out_channels=512))
        self.down_5 = nn.Sequential(conv_block(in_channels=512, out_channels=512),
                                    conv_block(in_channels=512, out_channels=512))

        self.middle = nn.Sequential(conv_block(in_channels=512, out_channels=512),
                                    conv_block(in_channels=512, out_channels=512))
        self.middle_t = conv_t_block(in_channels=512, out_channels=256)

        self.up_5 = nn.Sequential(conv_block(in_channels=768, out_channels=512),
                                  conv_block(in_channels=512, out_channels=512))
        self.up_5_t = conv_t_block(in_channels=512, out_channels=256)
        self.up_4 = nn.Sequential(conv_block(in_channels=768, out_channels=512),
                                  conv_block(in_channels=512, out_channels=512))
        self.up_4_t = conv_t_block(in_channels=512, out_channels=128)
        self.up_3 = nn.Sequential(conv_block(in_channels=384, out_channels=256),
                                  conv_block(in_channels=256, out_channels=256))
        self.up_3_t = conv_t_block(in_channels=256, out_channels=64)
        self.up_2 = nn.Sequential(conv_block(in_channels=192, out_channels=128),
                                  conv_block(in_channels=128, out_channels=128))
        self.up_2_t = conv_t_block(in_channels=128, out_channels=32)
        self.up_1 = nn.Sequential(conv_block(in_channels=96, out_channels=64),
                                  conv_block(in_channels=64, out_channels=1))

    def forward(self, x):
        down1 = self.down_1(x)  # (1x256x256 -> 64x256x256)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)  # (64x256x256 -> 64x128x128)

        down2 = self.down_2(out)  # (64x128x128 -> 128x128x128)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)  # (128x128x128 -> 128x64x64)

        down3 = self.down_3(out)  # (128x64x64 -> 256x64x64)
        out = F.max_pool2d(down3, kernel_size=2, stride=2)  # (256x64x64 -> 256x32x32)

        down4 = self.down_4(out)  # (256x32x32 -> 512x32x32)
        out = F.max_pool2d(down4, kernel_size=2, stride=2)  # (512x32x32 -> 512x16x16)

        down5 = self.down_5(out)  # (512x16x16 -> 512x16x16)
        out = F.max_pool2d(down5, kernel_size=2, stride=2)  # (512x16x16 -> 512x8x8)

        out = self.middle(out)  # (512x8x8 -> 512x8x8)
        out = self.middle_t(out, output_size=down5.size())  # (512x8x8 -> 256x16x16)

        out = torch.cat([down5, out], 1)  # (512x16x16-concat-256x16x16 -> 768x16x16)
        out = self.up_5(out)  # (768x16x16 -> 512x16x16)
        out = self.up_5_t(out, output_size=down4.size())  # (512x16x16 -> 256x32x32)

        out = torch.cat([down4, out], 1)  # (512x32x32-concat-256x32x32 -> 768x32x32)
        out = self.up_4(out)  # (768x32x32 -> 512x32x32)
        out = self.up_4_t(out, output_size=down3.size())  # (512x32x32 -> 128x64x64)

        out = torch.cat([down3, out], 1)  # (256x64x64-concat-128x64x64 -> 384x64x64)
        out = self.up_3(out)  # (384x64x64 -> 256x64x64)
        out = self.up_3_t(out, output_size=down2.size())  # (256x64x64 -> 64x128x128)

        out = torch.cat([down2, out], 1)  # (128x128x128-concat-64x128x128 -> 192x128x128)
        out = self.up_2(out)  # (192x128x128 -> 128x128x128)
        out = self.up_2_t(out, output_size=down1.size())  # (128x128x128 -> 32x256x256)

        out = torch.cat([down1, out], 1)  # (64x256x256-concat-32x256x256 -> 96x256x256)
        out = self.up_1(out)  # (96x256x256 -> 1x256x256)

        return out


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, scores, targets):
        scores_flat = scores.view(-1)
        targets_flat = targets.view(-1)
        return self.loss(scores_flat, targets_flat)


def box_mask(box, shape=1024):
    """
    :param box: [x, y, w, h] box coordinates
    :param shape: shape of the image (default set to maximum possible value, set to smaller to save memory)
    :returns: (np.array of bool) mask as binary 2D array
    """
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y + h, x:x + w] = True
    return mask


def parse_boxes(msk, threshold=0.20, connectivity=None):
    """
    :param msk: (torch.Tensor) CxWxH tensor representing the prediction mask
    :param threshold: threshold in the range 0-1 above which a pixel is considered a positive target
    :param connectivity: connectivity parameter for skimage.measure.label segmentation (can be None, 1, or 2)
                         http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    :returns: (list, list) predicted_boxes, confidences
    """
    # extract 2d array
    msk = msk[0]
    # select pixels above threshold and mark them as positives (1) in an array of equal size as the input prediction mask
    pos = np.zeros(msk.shape)
    pos[msk > threshold] = 1.
    # label regions
    lbl = skimage.measure.label(pos, connectivity=connectivity)

    predicted_boxes = []
    confidences = []
    # iterate over regions and extract box coordinates
    for region in skimage.measure.regionprops(lbl):
        # retrieve x, y, height and width and add to prediction string
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
        # add control over box size (eliminate if too small)
        if w * h > min_box_area:
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)

    return predicted_boxes, confidences


# define function that creates prediction strings as expected in submission
def prediction_string(predicted_boxes, confidences):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    """
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]


# define iou function
def IoU(pr, gt):
    """
    :param pr: (numpy_array(bool)) prediction array
    :param gt: (numpy_array(bool)) ground truth array
    :returns: IoU (pr, gt) = intersection (pr, gt) / union (pr, gt)
    """
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU


# define precision function
def precision(tp, fp, fn):
    """
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :returns: precision metric for one image at one threshold
    """
    return float(tp) / (tp + fp + fn + 1.e-9)


# define function that calculates the average precision of an image
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :param target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """

    # if both predicted and target boxes are empty, precision is NaN (and doesn't count towards the batch average)
    if predicted_boxes == [] and target_boxes == []:
        return np.nan
    else:
        # if we have predicted boxes but no target boxes, precision is 0
        if len(predicted_boxes) > 0 and target_boxes == []:
            return 0.0
        # if we have target boxes but no predicted boxes, precision is 0
        elif len(target_boxes) > 0 and predicted_boxes == []:
            return 0.0
        # if we have both predicted and target boxes, proceed to calculate image average precision
        else:
            # define list of thresholds for IoU [0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]
            thresholds = np.arange(0.4, 0.8, 0.05)
            # sort boxes according to their confidence (from largest to smallest)
            predicted_boxes_sorted = list(reversed([b for _, b in sorted(zip(confidences, predicted_boxes),
                                                                         key=lambda pair: pair[0])]))
            average_precision = 0.0
            for t in thresholds:  # iterate over thresholds
                # with a first loop we measure true and false positives
                tp = 0  # initiate number of true positives
                fp = len(predicted_boxes)  # initiate number of false positives
                for box_p in predicted_boxes_sorted:  # iterate over predicted boxes coordinates
                    box_p_msk = box_mask(box_p, shape)  # generate boolean mask
                    for box_t in target_boxes:  # iterate over ground truth boxes coordinates
                        box_t_msk = box_mask(box_t, shape)  # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk)  # calculate IoU
                        if iou > t:
                            tp += 1  # if IoU is above the threshold, we got one more true positive
                            fp -= 1  # and one less false positive
                            break  # proceed to the next predicted box
                # with a second loop we measure false negatives
                fn = len(target_boxes)  # initiate number of false negatives
                for box_t in target_boxes:  # iterate over ground truth boxes coordinates
                    box_t_msk = box_mask(box_t, shape)  # generate boolean mask
                    for box_p in predicted_boxes_sorted:  # iterate over predicted boxes coordinates
                        box_p_msk = box_mask(box_p, shape)  # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk)  # calculate IoU
                        if iou > t:
                            fn -= 1
                            break  # proceed to the next ground truth box
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision


# define function that calculates the average precision of a batch of images
def average_precision_batch(output_batch, pIds, pId_boxes_dict, rescale_factor, shape=1024, return_array=False):
    """
    :param output_batch: cnn model output batch
    :param pIds: (list) list of patient IDs contained in the output batch
    :param rescale_factor: CNN image rescale factor
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """

    batch_precisions = []
    for msk, pId in zip(output_batch, pIds):  # iterate over batch prediction masks and relative patient IDs
        # retrieve target boxes from dictionary (quicker than from mask itself)
        target_boxes = pId_boxes_dict[pId] if pId in pId_boxes_dict else []
        # rescale coordinates of target boxes
        if len(target_boxes) > 0:
            target_boxes = [[int(round(c / float(rescale_factor))) for c in box_t] for box_t in target_boxes]
        # extract prediction boxes and confidences
        predicted_boxes, confidences = parse_boxes(msk)
        batch_precisions.append(average_precision_image(predicted_boxes, confidences, target_boxes, shape=shape))
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions))


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, metric):
    """Saves model and training parameters at 'last.pth.tar'. If is_best==True, also saves
    'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
    """
    filename = 'last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, metric + '.best.pth.tar')


# train function
def train(model, dataloader, optimizer, loss_fn, pId_boxes_dict, rescale_factor, shape, save_summary_steps=5):
    model.train()
    summary = []
    loss_avg = RunningAverage()
    loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = [], [], []
    start = time.time()

    with tqdm(total=len(dataloader), desc="Training") as t:  # Use len(dataloader) for total
        for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device).float()
            labels_batch = labels_batch.to(device).float()
            optimizer.zero_grad()
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()
            loss_avg.update(loss.item())
            loss_t_hist_ep.append(loss.item())
            loss_avg_t_hist_ep.append(loss_avg())
            if i % save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                prec_batch = average_precision_batch(output_batch, pIds_batch, pId_boxes_dict, rescale_factor, shape)
                prec_t_hist_ep.append(prec_batch)
                summary_batch_string = "batch loss = {:05.7f} ;  ".format(loss.item())
                summary_batch_string += "average loss = {:05.7f} ;  ".format(loss_avg())
                summary_batch_string += "batch precision = {:05.7f} ;  ".format(prec_batch)
                t.set_postfix(loss=loss_avg())
                t.update()

            t.set_description(f'Batch {i}/{len(dataloader)}, time per {save_summary_steps} steps: {time.time() - start:.2f}s')

    metrics_string = "average loss = {:05.7f} ;  ".format(loss_avg())
    print("- Train epoch metrics summary: " + metrics_string)
    return loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep

# evaluate function
def evaluate(model, dataloader, loss_fn, pId_boxes_dict, rescale_factor, shape):
    model.eval()
    losses = []
    precisions = []
    start = time.time()
    
    with tqdm(total=len(dataloader), desc="Evaluating") as t:  # Use len(dataloader) for total
        for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device).float()
            labels_batch = labels_batch.to(device).float()
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, labels_batch)
            losses.append(loss.item())
            output_batch = output_batch.data.cpu()
            prec_batch = average_precision_batch(output_batch, pIds_batch, pId_boxes_dict, rescale_factor, shape, return_array=True)
            for p in prec_batch:
                precisions.append(p)
            t.update()

    metrics_mean = {'loss': np.nanmean(losses), 'precision': np.nanmean(np.asarray(precisions))}
    metrics_string = "average loss = {:05.7f} ;  ".format(metrics_mean['loss'])
    metrics_string += "average precision = {:05.7f} ;  ".format(metrics_mean['precision'])
    print("- Eval metrics : " + metrics_string)
    print('  Evaluation run in {:.2f} seconds.'.format(time.time() - start))

    return metrics_mean

# train_and_evaluate function (simplified to show main changes)
def train_and_evaluate(model, train_dataloader, val_dataloader, lr_init, loss_fn, num_epochs, pId_boxes_dict, rescale_factor, shape, restore_file=None):
    if restore_file is not None:
        checkpoint = torch.load(restore_file)
        model.load_state_dict(checkpoint['state_dict'])

    best_val_loss = 1e+15
    best_val_prec = 0.0
    best_loss_model = None
    best_prec_model = None

    loss_t_history = []
    loss_v_history = []
    loss_avg_t_history = []
    prec_t_history = []
    prec_v_history = []

    for epoch in range(num_epochs):
        start = time.time()

        # define the optimizer
        lr = lr_init * 0.5 ** float(epoch)  # reduce the learning rate at each epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Run one epoch
        print("Epoch {}/{}. Learning rate = {:05.3f}.".format(epoch + 1, num_epochs, lr))

        # train model for a whole epoch (one full pass over the training set)
        loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = train(model, train_dataloader, optimizer, loss_fn, pId_boxes_dict, rescale_factor, shape)
        loss_avg_t_history += loss_avg_t_hist_ep
        loss_t_history += loss_t_hist_ep
        prec_t_history += prec_t_hist_ep

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader, loss_fn, pId_boxes_dict, rescale_factor, shape)

        val_loss = val_metrics['loss']
        val_prec = val_metrics['precision']

        loss_v_history += len(loss_t_hist_ep) * [val_loss]
        prec_v_history += len(prec_t_hist_ep) * [val_prec]

        is_best_loss = val_loss <= best_val_loss
        is_best_prec = val_prec >= best_val_prec

        if is_best_loss:
            print("- Found new best loss: {:.4f}".format(val_loss))
            best_val_loss = val_loss
            best_loss_model = model
            # Save model immediately after finding a new best loss
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                            is_best=True,
                            metric='loss')

        if is_best_prec:
            print("- Found new best precision: {:.4f}".format(val_prec))
            best_val_prec = val_prec
            best_prec_model = model
            # Save model immediately after finding a new best precision
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                            is_best=True,
                            metric='prec')

        delta_time = time.time() - start
        print('Epoch run in {:.2f} minutes'.format(delta_time / 60.))

    histories = {'loss avg train': loss_avg_t_history,
                 'loss train': loss_t_history,
                 'precision train': prec_t_history,
                 'loss validation': loss_v_history,
                 'precision validation': prec_v_history}
    best_models = {'best loss model': best_loss_model,
                   'best precision model': best_prec_model}

    return histories, best_models



def predict(model, dataloader):
    # set model to evaluation mode
    model.eval()

    predictions = {}

    for i, (test_batch, pIds) in enumerate(tqdm(dataloader, desc="Predicting")):
        # Convert torch tensor to Variable
        test_batch = test_batch.to(device).float()

        # compute output
        output_batch = model(test_batch)
        sig = nn.Sigmoid().to(device)
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output

    return predictions

# define an instance of the model
model = PneumoniaUNET()
checkpoint = torch.load('last.pth.tar')
# Load the model state dictionary
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
# define the loss function
loss_fn = BCEWithLogitsLoss2d().to(device)
# define initial learning rate (will be reduced over epochs)
lr_init = 0.5

num_epochs = 5
num_steps_train = len(loader_train)
num_steps_eval = len(loader_valid)

shape = int(round(original_image_shape / rescale_factor))

# Train the model
print("Starting training for {} epochs".format(num_epochs))
histories, best_models = train_and_evaluate(model, loader_train, loader_valid, lr_init, loss_fn, 
                                            num_epochs, pId_boxes_dict, rescale_factor, shape)
# Print the histories for debugging
print("Training and evaluation histories:")
print(histories)

# If you want to save the best model for later use
torch.save(best_models['best loss model'].state_dict(), "best_loss_model.pth")
torch.save(best_models['best precision model'].state_dict(), "best_precision_model.pth")

# Create validation dataset for prediction
dataset_valid = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_valid, predict=True, 
                                 boxes=None, rescale_factor=rescale_factor, transform=transform)
loader_valid = DataLoader(dataset=dataset_valid,
                          batch_size=batch_size,
                          shuffle=False) 

def draw_boxes(predicted_boxes, confidences, target_boxes, ax, angle=0):
    if len(predicted_boxes) > 0:
        for box, c in zip(predicted_boxes, confidences):
            x, y, w, h = box 
            patch = Rectangle((x, y), w, h, color='red', ls='dashed',
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)
            ax.text(x + w / 2., y - 5, '{:.2}'.format(c), color='red', size=20, va='center', ha='center')
    if len(target_boxes) > 0:
        for box in target_boxes:
            x, y, w, h = box
            patch = Rectangle((x, y), w, h, color='red',  
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)
    return ax

def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x / rescale_factor))
    y = int(round(y / rescale_factor))
    w = int(round(w / rescale_factor))
    h = int(round(h / rescale_factor))
    return [x, y, w, h]    

predictions_valid = predict(best_models['best precision model'], loader_valid)
print('Predicted {} validation images.'.format(len(predictions_valid)))

best_threshold = None
best_avg_precision_valid = 0.0
thresholds = np.arange(0.01, 0.60, 0.01)
avg_precision_valids = []

for threshold in thresholds:
    precision_valid = []
    for i in tqdm(range(len(dataset_valid)), desc=f"Evaluating threshold {threshold:.2f}"):
        img, pId = dataset_valid[i]
        target_boxes = [rescale_box_coordinates(box, rescale_factor) for box in pId_boxes_dict[pId]] if pId in pId_boxes_dict else []
        prediction = predictions_valid[pId]
        predicted_boxes, confidences = parse_boxes(prediction, threshold=threshold, connectivity=None)
        avg_precision_img = average_precision_image(predicted_boxes, confidences, target_boxes, shape=img[0].shape[0])
        precision_valid.append(avg_precision_img)
    avg_precision_valid = np.nanmean(precision_valid)
    avg_precision_valids.append(avg_precision_valid)
    print('Threshold: {}, average precision validation: {:03.5f}'.format(threshold, avg_precision_valid))
    if avg_precision_valid > best_avg_precision_valid:
        print('Found new best average precision validation!')
        best_avg_precision_valid = avg_precision_valid
        best_threshold = threshold

# Create and save the submission file
best_model = best_models['best precision model']        
predictions_test = predict(best_model, loader_test)
print('Predicted {} images.'.format(len(predictions_test)))

df_sub = df_test[['patientId']].copy(deep=True)

def get_prediction_string_per_pId(pId):
    prediction = predictions_test[pId]
    predicted_boxes, confidences = parse_boxes(prediction, threshold=best_threshold, connectivity=None)
    predicted_boxes = [rescale_box_coordinates(box, 1 / rescale_factor) for box in predicted_boxes]
    return prediction_string(predicted_boxes, confidences)

df_sub['predictionString'] = df_sub['patientId'].apply(lambda x: get_prediction_string_per_pId(x) if x in pIds_test else '')
print('Number of non-null prediction strings: {} ({:05.2f}%)'.format(df_sub.loc[df_sub['predictionString'] != ''].shape[0],
                                                                     100. * df_sub.loc[df_sub['predictionString'] != ''].shape[0] / df_sub.shape[0]))
print(df_sub.head(10))
df_sub.to_csv('submission.csv', index=False)

