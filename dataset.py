import os
from skimage.transform import resize
import PIL
import numpy as np
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision.transforms.functional import to_pil_image
import pydicom
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()
from helper_fns import imgMinMaxScaler, elastic_transform

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