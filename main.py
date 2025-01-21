import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

from config import Config
from dataset import PneumoniaDataset
from Unet import PneumoniaUNET, BCEWithLogitsLoss2d
from helper_fns import *

validation_frac = Config.validation_frac
original_image_shape = Config.original_image_shape
datapath_prep = Config.datapath_prep
rescale_factor = Config.rescale_factor
batch_size = Config.batch_size
num_workers = Config.num_workers
transform = Config.transform
device = Config.device

df_train = pd.read_csv(Config.datapath_prep + 'train.csv')

df_test = pd.read_csv(Config.datapath_prep + 'test.csv')
print(df_train.head(3))

df_train = df_train.sample(frac=1, random_state=42) 
pIds = [pId for pId in df_train['patientId'].unique()]

pIds_valid = pIds[: int(round(validation_frac * len(pIds)))]
pIds_train = pIds[int(round(validation_frac * len(pIds))):]

print('{} patient IDs shuffled and {}% of them used in validation set.'.format(len(pIds), validation_frac * 100))
print('{} images went into train set and {} images went into validation set.'.format(len(pIds_train), len(pIds_valid)))


pIds_test = df_test['patientId'].unique()
print('{} patient IDs in test set.'.format(len(pIds_test)))


pId_boxes_dict = {}
for pId in df_train.loc[(df_train['Target'] == 1)]['patientId'].unique().tolist():
    pId_boxes_dict[pId] = get_boxes_per_patient(df_train, pId)
print('{} ({:.1f}%) images have target boxes.'.format(len(pId_boxes_dict), 100 * (len(pId_boxes_dict) / len(pIds))))


dataset_train = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_train, predict=False,
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=3, warping=True)

dataset_valid = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_valid, predict=False,
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle=0, warping=False)

dataset_test = PneumoniaDataset(root=datapath_prep, subset='test', pIds=pIds_test, predict=True,
                                boxes=None, rescale_factor=rescale_factor, transform=transform,
                                rotation_angle=0, warping=False)


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

def train(model, dataloader, optimizer, loss_fn, pId_boxes_dict, rescale_factor, shape, save_summary_steps=5):
    model.train()
    summary = []
    loss_avg = RunningAverage()
    loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = [], [], []
    start = time.time()

    with tqdm(total=len(dataloader), desc="Training") as t:  
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


def evaluate(model, dataloader, loss_fn, pId_boxes_dict, rescale_factor, shape):
    model.eval()
    losses = []
    precisions = []
    start = time.time()
    
    with tqdm(total=len(dataloader), desc="Evaluating") as t:
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

        
        lr = lr_init * 0.5 ** float(epoch)  # reduce the learning rate at each epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       
        print("Epoch {}/{}. Learning rate = {:05.3f}.".format(epoch + 1, num_epochs, lr))

        
        loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = train(model, train_dataloader, optimizer, loss_fn, pId_boxes_dict, rescale_factor, shape)
        loss_avg_t_history += loss_avg_t_hist_ep
        loss_t_history += loss_t_hist_ep
        prec_t_history += prec_t_hist_ep

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
    model.eval()

    predictions = {}

    for i, (test_batch, pIds) in enumerate(tqdm(dataloader, desc="Predicting")):
        test_batch = test_batch.to(device).float()

        output_batch = model(test_batch)
        sig = nn.Sigmoid().to(device)
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output

    return predictions

model = PneumoniaUNET()
checkpoint = torch.load('last.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
loss_fn = BCEWithLogitsLoss2d().to(device)
lr_init = 0.5

num_epochs = 5
num_steps_train = len(loader_train)
num_steps_eval = len(loader_valid)

shape = int(round(original_image_shape / rescale_factor))

print("Starting training for {} epochs".format(num_epochs))
histories, best_models = train_and_evaluate(model, loader_train, loader_valid, lr_init, loss_fn, 
                                            num_epochs, pId_boxes_dict, rescale_factor, shape)

print("Training and evaluation histories:")
print(histories)

torch.save(best_models['best loss model'].state_dict(), "best_loss_model.pth")
torch.save(best_models['best precision model'].state_dict(), "best_precision_model.pth")


dataset_valid = PneumoniaDataset(root=datapath_prep, subset='train', pIds=pIds_valid, predict=True, 
                                 boxes=None, rescale_factor=rescale_factor, transform=transform)
loader_valid = DataLoader(dataset=dataset_valid,
                          batch_size=batch_size,
                          shuffle=False) 
 
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

