import skimage
from scipy.ndimage import map_coordinates, gaussian_filter
import torch
import numpy as np
from matplotlib.patches import Rectangle
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

from config import Config

def get_boxes_per_patient(df, pId):
    boxes = df.loc[df['patientId'] == pId][['x', 'y', 'width', 'height']].astype('int').values.tolist()
    return boxes
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
        if w * h > Config.min_box_area:
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


def IoU(pr, gt):
    """
    :param pr: (numpy_array(bool)) prediction array
    :param gt: (numpy_array(bool)) ground truth array
    :returns: IoU (pr, gt) = intersection (pr, gt) / union (pr, gt)
    """
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU


def precision(tp, fp, fn):
    """
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :returns: precision metric for one image at one threshold
    """
    return float(tp) / (tp + fp + fn + 1.e-9)


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
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, metric):
    filename = 'last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, metric + '.best.pth.tar')
        
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

