import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
import os
import copy

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import csv

import scipy.stats as ss

from sklearn import svm, datasets
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from functools import partial


from .MODEL_ARCHITECTURE import FullModel
from .LOSS_METRICS import LogProbLoss, MSELoss, MAELoss
from .DATASETCLASS import LightCurvesDataset, collate_lcs

def create_split_folders(train_folder='./dataset/train/', test_folder='./dataset/test/', val_folder='./dataset/val/'):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

def prepare_output_dir(OUTPUT_PATH):
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for name in files:
            os.remove(os.path.join(root, name))


def split_data(files, DATA_SRC, TRAIN_FOLDER, TEST_FOLDER, VAL_FOLDER, split_lists=None, verbose=0):
    # If custom split lists are provided, use them; otherwise, split randomly
    if split_lists is None:
        random.shuffle(files)
        val_files = random.sample(files, 2)  # Pick 2 files for validation
    else:
        train_files, test_files, val_files = split_lists

    i = 0
    train_list, test_list, val_list = [], [], []

    # Determine which file list to use (tqdm for progress display if verbose > 0)
    files_to_use = tqdm(files) if verbose > 0 else files

    for file in files_to_use:
        lcName = file.split(".")[0]
        tmpDataFrame = pd.read_csv(os.path.join(DATA_SRC, file))

        if split_lists is None:  # Random split if no custom lists are provided
            if file in val_files:
                filename = VAL_FOLDER + lcName + '_split' + str(i) + '.csv'
                val_list.append(lcName)
            else:
                r = random.uniform(0, 1)
                if r < 0.8:
                    train_list.append(lcName)
                    filename = TRAIN_FOLDER + lcName + '_split' + str(i) + '.csv'
                elif r < 0.9:
                    test_list.append(lcName)
                    filename = TEST_FOLDER + lcName + '_split' + str(i) + '.csv'
                else:
                    val_list.append(lcName)
                    filename = VAL_FOLDER + lcName + '_split' + str(i) + '.csv'
        else:  # Use custom split lists
            if lcName in train_files:
                train_list.append(lcName)
                filename = TRAIN_FOLDER + lcName + '_split' + str(i) + '.csv'
            elif lcName in test_files:
                test_list.append(lcName)
                filename = TEST_FOLDER + lcName + '_split' + str(i) + '.csv'
            elif lcName in val_files:
                val_list.append(lcName)
                filename = VAL_FOLDER + lcName + '_split' + str(i) + '.csv'
        tmpDataFrame.to_csv(filename, index=False)
        i += 1

    return train_list, test_list, val_list

torch.cuda.empty_cache() 

# REPRODUCIBILITY  
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_data_loaders(data_path_train, data_path_val, batch_size,tf_dir = None,param_df = None, param_columns = None, class_labels_df = None, data_type = 'train',augment = True,num_workers = None):
    if num_workers is None:
        num_workers = max(0, os.cpu_count() // 2)

    
    if data_type == 'train':
        train_set = LightCurvesDataset(root_dir=data_path_train, status='train',tf_dir = tf_dir,param_df = param_df, param_columns = param_columns, class_labels_df = class_labels_df)
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=partial(collate_lcs, augment = augment),
                                  num_workers=0,
                                  pin_memory=True,
                                  persistent_workers = False)
        return train_loader
    elif data_type == 'val':
        val_set = LightCurvesDataset(root_dir=data_path_val, status='test',tf_dir = tf_dir,param_df = param_df, param_columns = param_columns, class_labels_df = class_labels_df)
        val_loader = DataLoader(val_set,
                                num_workers=0,
                                batch_size=1,
                                pin_memory=True,
                                persistent_workers = False)

        return val_loader


def create_model_and_optimizer(device,encoding_size,latent_dim,attention,self_attention,latent_mlp_size = 128,attention_type = 'scaledot',\
                              no_latent_space_sample = 1,latent_mode = 'NPVI',lstm_layers = 0,lr = 1e-4,lstm_size = 64,activation = 'relu',lstm_agg = False,use_scheduler = True,transfer_function_length = 0, parameters_length = 0, classes = 0, replace_lstm_with_gru = False, bidirectional = False,num_workers = None):
    if latent_dim is None:
        no_latent_space_sample = None
    model = FullModel(encoding_size,latent_dim,latent_mlp_size,attention,self_attention,attention_type,no_latent_space_sample,latent_mode,lstm_layers,\
                     lstm_size,activation,lstm_agg,transfer_function_length, parameters_length, classes,replace_lstm_with_gru = replace_lstm_with_gru,bidirectional = bidirectional)
    model = model.to(device,non_blocking = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        scheduler = None
    criterion = LogProbLoss(no_latent_space_sample)
    mseMetric = MSELoss()
    maeMetric = MAELoss()
    
    return model, optimizer, scheduler, criterion, mseMetric, maeMetric

def track_gradients(model):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print('Total Gradient Norm:', total_norm)

def train_model(model, criterion, optimizer, scheduler, num_runs, EPOCHS, EARLY_STOPPING_LIMIT, mseMetric, maeMetric, device,DATA_PATH_TRAIN, \
                DATA_PATH_VAL, BATCH_SIZE,latent_mode = 'NPVI',beta_param = 0, beta_tf = 0, beta_classifier = 0,tf_dir = None,param_df=None,\
                param_columns=None,class_labels_df = None,augment = True,validation_epochs = 10,num_workers = 0):
    history_loss_train = [[] for _ in range(num_runs)]
    history_loss_val = [[] for _ in range(num_runs)]
    #The loss from parameters
    history_loss_tf_train = [[] for _ in range(num_runs)]
    history_loss_tf_val = [[] for _ in range(num_runs)]
    history_loss_param_train = [[] for _ in range(num_runs)]
    history_loss_param_val = [[] for _ in range(num_runs)]
    history_loss_classes_train = [[] for _ in range(num_runs)]
    history_loss_classes_val = [[] for _ in range(num_runs)]
    #The kl loss
    history_kl_loss_train = [[] for _ in range(num_runs)]
    history_kl_loss_val = [[] for _ in range(num_runs)]
    #The reconstruction loss
    history_loss_reconstruction_train = [[] for _ in range(num_runs)]
    history_loss_reconstruction_val = [[] for _ in range(num_runs)]
    #Other losses
    history_mse_train = [[] for _ in range(num_runs)]
    history_mse_val = [[] for _ in range(num_runs)]
    history_mae_train = [[] for _ in range(num_runs)]
    history_mae_val = [[] for _ in range(num_runs)]
    #Epoch counters
    epoch_counter_train_loss = [[] for _ in range(num_runs)]
    epoch_counter_train_mse = [[] for _ in range(num_runs)]
    epoch_counter_train_mae = [[] for _ in range(num_runs)]
    epoch_counter_val_loss = [[] for _ in range(num_runs)]
    epoch_counter_val_mse = [[] for _ in range(num_runs)]
    epoch_counter_val_mae = [[] for _ in range(num_runs)]
    
    criterion_tf = LogProbLoss(None,param = True)
    criterion_classifier = nn.CrossEntropyLoss()
    criterion_param = LogProbLoss(None,param = True)
    
    valLoader = get_data_loaders(DATA_PATH_TRAIN, DATA_PATH_VAL, BATCH_SIZE,tf_dir,param_df, param_columns, class_labels_df,data_type = 'val',num_workers = num_workers)
    
    for j in range(num_runs):
        epochs_since_last_improvement = 0
        best_loss = None
        best_model = copy.deepcopy(model.state_dict())
        epoch_counter = 0

        for epoch in tqdm(range(EPOCHS)):
            epoch_counter = epoch + 1
            model.train()
            total_loss_train = 0
            total_mse_train = 0
            total_mae_train = 0
            total_loss_reconstruction_train = 0
            total_loss_tf_train = 0
            total_loss_kl_train = 0
            total_loss_param_train = 0
            total_loss_classifier_train = 0
            counter = 0

            trainLoader = get_data_loaders(DATA_PATH_TRAIN, DATA_PATH_VAL, BATCH_SIZE,tf_dir,param_df, param_columns, class_labels_df,data_type = 'train',augment = augment,num_workers = num_workers)

            for data in trainLoader:
                # Unpack data
                [context_x, context_y, target_x, measurement_error], target_y,tf,param,class_labels = data

                # Move to GPU
                context_x, context_y, target_x, target_y, measurement_error = (
                    context_x.to(device,non_blocking=True),
                    context_y.to(device,non_blocking=True),
                    target_x.to(device,non_blocking=True),
                    target_y.to(device,non_blocking=True),
                    measurement_error.to(device,non_blocking=True),
                )
                
                if param != 'None':
                    param = param.to(device,non_blocking=True)
                if tf != 'None':
                    tf = tf.to(device,non_blocking=True)
                if class_labels != 'None':
                    class_labels = class_labels.to(device,non_blocking=True)
                
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                if latent_mode == 'NPVI':
                    target_y_dummy = target_y
                else:
                    target_y_dummy = None
                dist, mu, sigma, kl_loss, z, R, latent_dist, agg_R_z, predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_x,target_y_dummy)
                
                if kl_loss is not None:
                    total_loss_kl_train += torch.mean(kl_loss).item()
                else:
                    total_loss_kl_train += 0
                
                # Calculate loss and do a backward pass
                loss_mag = criterion(dist, target_y,kl_loss)
                total_loss_reconstruction_train += loss_mag.item()
                
                #Whether to consider parameters or not
                if predicted_parameters is not None:
                    loss_param = criterion_param(predicted_parameters,param)
                    total_loss_param_train += loss_param.item()
                else:
                    loss_param = 0
                    total_loss_param_train += 0
                
                if predicted_tf is not None:
                    loss_tf = criterion_tf(predicted_tf,tf)
                    total_loss_tf_train += loss_tf.item()
                else:
                    loss_tf = 0
                    total_loss_tf_train += 0
                
                    
                if predicted_classes is not None:
                    loss_classes = criterion_classifier(predicted_classes.mean(dim = 0),class_labels)
                    total_loss_classifier_train += loss_classes.item()
                else:
                    loss_classes = 0
                    total_loss_classifier_train += 0
                    
                loss = loss_mag + beta_param*loss_param + beta_tf*loss_tf + beta_classifier*loss_classes
                counter += 1
                total_loss_train += loss.item()
                loss.backward()

                # Update weights
                optimizer.step()
                
                if len(mu.shape) == 3:
                    mse_mae_mu = mu.mean(0)
                else:
                    mse_mae_mu = mu
                
                # Calculate MSE metric (use the error as an inverse weight)
                mseLoss = mseMetric(target_y, mse_mae_mu, 1/(measurement_error)**2)
                total_mse_train += mseLoss.item()

                # Calculate MAE metric (use the error as an inverse weight)
                maeLoss = maeMetric(target_y, mse_mae_mu,  1/(measurement_error)**2)
                total_mae_train += maeLoss.item()

            # Update history for losses
            epoch_loss = total_loss_train / len(trainLoader)
            history_loss_train[j].append(epoch_loss)
            
            epoch_reconstruct_loss = total_loss_reconstruction_train / len(trainLoader)
            history_loss_reconstruction_train[j].append(epoch_reconstruct_loss)
            
            epoch_loss_tf = total_loss_tf_train / len(trainLoader)
            history_loss_tf_train[j].append(epoch_loss_tf)
            
            epoch_loss_param = total_loss_param_train / len(trainLoader)
            history_loss_param_train[j].append(epoch_loss_param)
            
            epoch_loss_classifier = total_loss_classifier_train / len(trainLoader)
            history_loss_classes_train[j].append(epoch_loss_classifier)
            
            epoch_kl_loss = total_loss_kl_train / len(trainLoader)
            history_kl_loss_train[j].append(epoch_kl_loss)

            epoch_mse = total_mse_train / len(trainLoader)
            history_mse_train[j].append(epoch_mse)

            epoch_mae = total_mae_train / len(trainLoader)
            history_mae_train[j].append(epoch_mae)

            if epoch_counter%validation_epochs == 0:
                # Validation
                model.eval()
                with torch.no_grad():
                    total_loss_val = 0
                    total_mse_val = 0
                    total_mae_val = 0
                    total_loss_reconstruction_val = 0
                    total_loss_tf_val = 0
                    total_loss_kl_val = 0
                    total_loss_param_val = 0
                    total_loss_classifier_val = 0
                    for data in valLoader:
                        # Unpack data
                        context_x, context_y, target_x, target_y, target_test_x, measurement_error = data[
                            "context_x"
                        ], data["context_y"], data["target_x"], data["target_y"], data["target_test_x"], data[
                            "measurement_error"
                        ]
                        
                        tf,param,class_labels = data['transfer_function'],data['parameters'],data['tf_class_labels']
                        
    
                        # Move to GPU
                        context_x, context_y, target_x, target_y, target_test_x, measurement_error = (
                            context_x.to(device,non_blocking=True),
                            context_y.to(device,non_blocking=True),
                            target_x.to(device,non_blocking=True),
                            target_y.to(device,non_blocking=True),
                            target_test_x.to(device,non_blocking=True),
                            measurement_error.to(device,non_blocking=True),
                        )
                        
                        if param[0] != 'None':
                            param = param.to(device,non_blocking=True)
                        if tf[0] != 'None':
                            tf = tf.to(device,non_blocking=True)
                        if class_labels[0] != 'None':
                            class_labels = class_labels.to(device,non_blocking=True)
                        
                        
                        if latent_mode == 'NPVI':
                            target_y_dummy = target_y
                        else:
                            target_y_dummy = None
    
                        # Forward Pass
                        dist, mu, sigma, kl_loss, z, R, latent_dist, agg_R_z, predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_x,target_y_dummy)
                        
                        if kl_loss is not None:
                            total_loss_kl_val += torch.mean(kl_loss).item()
                        else:
                            total_loss_kl_val += 0
    
                        # Calculate loss
                        loss_mag = criterion(dist, target_y,kl_loss)
                        total_loss_reconstruction_val += loss_mag.item()
    
                        #Whether to consider parameters or not
                        if predicted_parameters is not None:
                            loss_param = criterion_param(predicted_parameters,param)
                            total_loss_param_val += loss_param.item()
                        else:
                            loss_param = 0
                            total_loss_param_val += 0
    
                        if predicted_tf is not None:
                            loss_tf = criterion_tf(predicted_tf,tf)
                            total_loss_tf_val += loss_tf.item()
                        else:
                            loss_tf = 0
                            total_loss_tf_val += 0
    
                        if predicted_classes is not None:
                            loss_classes = criterion_classifier(predicted_classes.mean(dim=0),class_labels)
                            total_loss_classifier_val += loss_classes.item()
                        else:
                            loss_classes = 0
                            total_loss_classifier_val += 0
    
                        loss = loss_mag + beta_param*loss_param + beta_tf*loss_tf + beta_classifier*loss_classes
                        total_loss_val += loss.item()

                                        
                        if len(mu.shape) == 3:
                            mse_mae_mu = mu.mean(0)
                        else:
                            mse_mae_mu = mu

                        # Calculate MSE metric (use the error as an inverse weight)
                        mseLossval = mseMetric(target_y, mse_mae_mu, 1/(measurement_error)**2)
                        total_mse_val += mseLossval.item()

                        # Calculate MAE metric (use the error as an inverse weight)
                        maeLossval = maeMetric(target_y, mse_mae_mu,  1/(measurement_error)**2)
                        total_mae_val += maeLossval.item()
                        
                
                # Update history for losses
                val_loss = total_loss_val / len(valLoader)
                history_loss_val[j].append(val_loss)
                
                val_reconstruct_loss = total_loss_reconstruction_val / len(valLoader)
                history_loss_reconstruction_val[j].append(val_reconstruct_loss)
                
                val_loss_tf = total_loss_tf_val / len(valLoader)
                history_loss_tf_val[j].append(val_loss_tf)
                
                val_loss_param = total_loss_param_val / len(valLoader)
                history_loss_param_val[j].append(val_loss_param)
                
                val_loss_classifier = total_loss_classifier_val / len(valLoader)
                history_loss_classes_val[j].append(val_loss_classifier)
                
                val_kl_loss = total_loss_kl_val / len(valLoader)
                history_kl_loss_val[j].append(val_kl_loss)
    
                val_mse = total_mse_val / len(valLoader)
                history_mse_val[j].append(val_mse)
    
                val_mae = total_mae_val / len(valLoader)
                history_mae_val[j].append(val_mae)
    
                # Early stopping
                if best_loss is None:
                    best_loss = val_loss
                            
                if scheduler is not None:
                    scheduler.step()
    
                if val_loss >= best_loss:
                    epochs_since_last_improvement += validation_epochs
                    if epochs_since_last_improvement >= EARLY_STOPPING_LIMIT:
                        print(f"Early stopped at epoch {epoch}!")
                        print(f"Best model at epoch {epoch - epochs_since_last_improvement}!")
                        model.load_state_dict(best_model)
                        break
                else:
                    epochs_since_last_improvement = 0
                    best_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict())
                epoch_counter_val_loss[j].append(epoch_counter)
                epoch_counter_val_mse[j].append(epoch_counter)
                epoch_counter_val_mae[j].append(epoch_counter)

            epoch_counter_train_loss[j].append(epoch_counter)
            epoch_counter_train_mse[j].append(epoch_counter)
            epoch_counter_train_mae[j].append(epoch_counter)

    return (
        #The full losses
        history_loss_train,
        history_loss_val,
        #The mse losses
        history_mse_train,
        history_mse_val,
        #The mae losses
        history_mae_train,
        history_mae_val,
        #Epoch Counters
        epoch_counter_train_loss,
        epoch_counter_train_mse,
        epoch_counter_train_mae,
        epoch_counter_val_loss,
        epoch_counter_val_mse,
        epoch_counter_val_mae,
        #Reconstruction Loss for the curves
        history_loss_reconstruction_train,
        history_loss_reconstruction_val,
        #Transfer Function Loss
        history_loss_tf_train,
        history_loss_tf_val,
        #Param Loss
        history_loss_param_train,
        history_loss_param_val,
        #Classifier Loss
        history_loss_classes_train,
        history_loss_classes_val,
        #KL Loss
        history_kl_loss_train,
        history_kl_loss_val,
    )

def save_lists_to_csv(file_names, lists):
    for file_name, data_list in zip(file_names, lists):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data_list:
                writer.writerow(row)


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_loss(history_loss_train_file, history_loss_val_file, epoch_counter_train_loss_file):
    history_loss_train = np.loadtxt(history_loss_train_file, delimiter=',')
    history_loss_val = np.loadtxt(history_loss_val_file, delimiter=',')
    epoch_counter_train_loss = np.loadtxt(epoch_counter_train_loss_file, delimiter=',')
    
    history_loss_train = history_loss_train[:len(epoch_counter_train_loss)]
    history_loss_val = history_loss_val[:len(epoch_counter_train_loss)]

   
    plt.plot(epoch_counter_train_loss, history_loss_train, label='Train LOSS')
    plt.plot(epoch_counter_train_loss, history_loss_val, label='Validation LOSS')
    plt.title("LogProbLOSS")
    plt.xlabel("Epoch")
    plt.ylabel("LOSS")
    plt.legend()
    plt.show()

def plot_mse(history_mse_train_file, history_mse_val_file, epoch_counter_train_mse_file):
    history_mse_train = np.loadtxt(history_mse_train_file, delimiter=',')
    history_mse_val = np.loadtxt(history_mse_val_file, delimiter=',')
    epoch_counter_train_mse = np.loadtxt(epoch_counter_train_mse_file, delimiter=',')

    history_mse_train = history_mse_train[:len(epoch_counter_train_mse)]
    history_mse_val = history_mse_val[:len(epoch_counter_train_mse)]

    epoch_counter = len(epoch_counter_train_mse)
    plt.plot(epoch_counter_train_mse, history_mse_train, label='Train MSE')
    plt.plot(epoch_counter_train_mse, history_mse_val, label='Validation MSE')
    plt.title("Mean Squared Error (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_mae(history_mae_train_file, history_mae_val_file, epoch_counter_train_mae_file):
    history_mae_train = np.loadtxt(history_mae_train_file, delimiter=',')
    history_mae_val = np.loadtxt(history_mae_val_file, delimiter=',')
    epoch_counter_train_mae = np.loadtxt(epoch_counter_train_mae_file, delimiter=',')

    history_mae_train = history_mae_train[:len(epoch_counter_train_mae)]
    history_mae_val = history_mae_val[:len(epoch_counter_train_mae)]

    epoch_counter = len(epoch_counter_train_mae)
    plt.plot(epoch_counter_train_mae, history_mae_train, label='Train MAE')
    plt.plot(epoch_counter_train_mae, history_mae_val, label='Validation MAE')
    plt.title("Mean Absolute Error (MAE)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()


def save_model(model, MODEL_PATH):
    torch.save(model.state_dict(), MODEL_PATH)
