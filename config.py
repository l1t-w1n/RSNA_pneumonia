import torch
import torchvision as tv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_image_shape = 1024
    datapath_prep = "./data/"
    
    validation_frac = 0.10
    
    rescale_factor = 4
    batch_size = 10  
    num_workers = 20  
    
    min_box_area = 10000
    min_box_area = int(round(min_box_area / float(rescale_factor ** 2)))

    #add normalization of images into transforms
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])