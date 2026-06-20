import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import collections
import numpy as np
import math

import random
import csv
from datetime import datetime

from cycler import cycler

import os
import glob

from tqdm import tqdm

import json

import dill
import pickle

import math

import pandas as pd

from sklearn import svm, datasets
import scipy.stats as ss

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error


from .MODEL_ARCHITECTURE import FullModel
from .LOSS_METRICS import LogProbLoss, MSELoss
from .DATASETCLASS import LightCurvesDataset, collate_lcs

tf_loss = LogProbLoss(None,param = True)
param_loss = LogProbLoss(None,param = True)

def create_prediction_folders(base_dir='./output/predictions',verbose = 0):
    sets = ['train', 'test', 'val']
    subfolders = ['plots', 'data']

    for set_folder in sets:
        set_path = os.path.join(base_dir, set_folder)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
            if verbose == 1:
                print(f"Created folder: {set_path}")
        else:
            if verbose == 1:
                print(f"Folder already exists: {set_path}")

        for subfolder in subfolders:
            subfolder_path = os.path.join(set_path, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                if verbose == 1:
                    print(f"Created folder: {subfolder_path}")
            else:
                if verbose == 1:
                    print(f"Folder already exists: {subfolder_path}")



def prepare_output_dir(OUTPUT_PATH):
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for name in files:
            os.remove(os.path.join(root, name))
            
            
def load_trained_model(MODEL_PATH, device,encoding_size,latent_dim,latent_mlp_size,attention,self_attention,attention_type = 'scaledot',\
                              no_latent_space_sample = 1,latent_mode = 'NPVI',lstm_layers = 0,lstm_size = 64,activation = 'relu',lstm_agg = False,transfer_function_length = 0, parameters_length = 0, classes = 0,replace_lstm_with_gru = True,bidirectional = False):
    if latent_dim is None:
        no_latent_space_sample = None
    model = FullModel(encoding_size,latent_dim,latent_mlp_size,\
                      attention,self_attention,attention_type,no_latent_space_sample,latent_mode,lstm_layers,\
                     lstm_size,activation,lstm_agg,transfer_function_length, parameters_length, classes,replace_lstm_with_gru,bidirectional)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()    
    return model


def get_criteria(no_latent_space_sample = 1):
    criterion = LogProbLoss(no_latent_space_sample)
    mseMetric = MSELoss()
    
    return criterion, mseMetric

def remove_padded_values_and_filter(folder_path,verbose = 0):
    # Get the list of CSV files in the input folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

    # Iterate over the CSV files
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)

        # Check if the filename contains "minus" or "plus"
        if "minus" in filename or "plus" in filename:
            # Delete the CSV file
            os.remove(file_path)
            if verbose == 1:
                print(f"Deleted file with 'minus' or 'plus' in the name: {filename}")
        else:
            # Remove padded values from the curve and keep the original
            try:
                # Read the CSV file and load the data into a pandas DataFrame
                data = pd.read_csv(file_path)

                # Check if the DataFrame has more than 1 row
                if len(data) > 1:
                    # Check if 'mag' and 'magerr' columns are the same as the last observation
                    last_row = data.iloc[-1]
                    mag_values = data['cont']
                    magerr_values = data['conterr']
                    if not (mag_values == last_row['cont']).all() or not (magerr_values == last_row['conterr']).all():
                        # Keep rows where 'mag' and 'magerr' are not the same as the last observation
                        data = data[(mag_values != last_row['cont']) | (magerr_values != last_row['conterr'])]

                        # Overwrite the original CSV file with the modified DataFrame
                        data.to_csv(file_path, index=False)
                        if verbose == 1:
                            print(f"Removed padding in file: {filename}")
                    else:
                        print(f"No padding removed for file: {filename}")
                else:
                    print(f"No padding removed for file: {filename}")

            except pd.errors.EmptyDataError:
                print(f"Error: Empty file encountered: {filename}")


def load_trcoeff(PATH):
    with open(PATH, "rb") as f:
        tr = dill.load(f)
    return tr

def back_x(x,Ax,Bx):
  a=-2
  b=2
  gornji=Bx-Ax
  donji=b-a
  xorig=Ax+((x-a)*(gornji/donji))
  return xorig

def back_y(y,Ay,By):
  a=-2
  b=2
  gornji=By-Ay
  donji=b-a
  yorig=Ay+((y-a)*(gornji/donji))
  return yorig

OUTPUT_PATH="./output/predictions/"
    
###NEW PLOT FUNCTION
def plot_function2(tr,target_x, target_y, context_x, context_y, yerr1, pred_y, var, target_test_x, lcName, OUTPUT_PATH, save = False, isTrainData = None, flagval = 0, notTrainData = None, mu_target = None, sigma_target = None,cut_little = False):
    """Plots the light curve data and predicted mean and variance.

    Args: 
    context_x: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the x values of the context points.
    context_y: Array of shape BATCH_SIZE x NUM_CONTEXT that contains the y values of the context points.
    target_x: Array of shape BATCH_SIZE x NUM_TARGET that contains the x values of the target points.
    target_y: Array of shape BATCH_SIZE x NUM_TARGET that contains the ground truth y values of the target points.
    target_test_x: Array of shape BATCH_SIZE x 400 that contains uniformly spread points across in [-2, 2] range.
    yerr1: Array of shape BATCH_SIZE x NUM_measurement_error that contains the measurement errors.
    pred_y: Array of shape BATCH_SIZE x 400 that contains predictions across [-2, 2] range.
    var: An array of shape BATCH_SIZE x 400  that contains the variance of predictions at target_test_x points.
    tr: array of data in pickle format needed to backtransform data from [-2,2] x [-2,2] to MJD x Mag
    CI: The confidence interval to plot
    """
    # Move to cpu
    target_x, target_y, context_x, context_y,yerr1, pred_y, var = target_x.cpu(), target_y.cpu(), \
                                                              context_x.cpu(), context_y.cpu(), yerr1.cpu(),\
                                                              pred_y.cpu(), var.cpu()
    if mu_target is not None and sigma_target is not None:
        mu_target = mu_target.cpu()
        sigma_target = sigma_target.cpu()
    
    lcName_no = lcName.split('_split')[0]
    tr = next((sublist for sublist in tr if sublist[0] == lcName_no),None)

    target_test_x = target_test_x.cpu()
        
    target_test_xorig=back_x(target_test_x[0].numpy(),tr[1],tr[2])
    
    #Plot differently if using different number of latent space samples
    pred_yorig = []
    vartop_1 = []
    varbot_1 = []
    vartop_2 = []
    varbot_2 = []
    vartop_3 = []
    varbot_3 = []
    for i in range(pred_y.shape[0]): 
        pred_yorig.append(back_y(pred_y[i][0].numpy(),tr[3],tr[4]))
        #The top and bottom errors are transformed seperately
        #3 different samples are used to get the different sigma intervals
        vartop_1.append(back_y(pred_y[i][0].numpy()+var[i][0,:].numpy(),tr[3],tr[4]))
        varbot_1.append(back_y(pred_y[i][0].numpy()-var[i][0,:].numpy(),tr[3],tr[4]))
        vartop_2.append(back_y(pred_y[i][0].numpy()+2*var[i][0,:].numpy(),tr[3],tr[4]))
        varbot_2.append(back_y(pred_y[i][0].numpy()-2*var[i][0,:].numpy(),tr[3],tr[4]))
        vartop_3.append(back_y(pred_y[i][0].numpy()+3*var[i][0,:].numpy(),tr[3],tr[4]))
        varbot_3.append(back_y(pred_y[i][0].numpy()-3*var[i][0,:].numpy(),tr[3],tr[4]))
    pred_yorig = np.array(pred_yorig)
    vartop_1 = np.array(vartop_1)
    varbot_1 = np.array(varbot_1)
    vartop_2 = np.array(vartop_2)
    varbot_2 = np.array(varbot_2)
    vartop_3 = np.array(vartop_3)
    varbot_3 = np.array(varbot_3)
    
    #Transform the target points from the observations also
    pred_y_observed = []
    predicted_var_observed_top = []
    predicted_var_observed_bot = []
    if mu_target is not None and sigma_target is not None:
        for i in range(pred_y.shape[0]):
            pred_y_observed.append(back_y(mu_target[i][0].numpy(),tr[3],tr[4]))
            predicted_var_observed_top.append(back_y(mu_target[i][0].numpy()+sigma_target[i][0,:].numpy(),tr[3],tr[4]))
            predicted_var_observed_bot.append(back_y(mu_target[i][0].numpy()-sigma_target[i][0,:].numpy(),tr[3],tr[4]))
    
    #Transform the original target points
    target_xorig=back_x(target_x[0].numpy(),tr[1],tr[2])
    target_yorig = back_y(target_y[0].numpy(),tr[3],tr[4])
    
    # Plot everything
    for i in range(len(pred_yorig)):
        if i == 0: 
            label_smooth = 'mean model: {} samples'.format(len(pred_yorig))
            label1sig = r'1$\sigma$'
            label2sig = r'2$\sigma$'
            label3sig = r'3$\sigma$'
            label_target = 'Target Predictions'
        else:
            label_smooth = None
            label1sig = None
            label2sig = None
            label3sig = None
            label_target = None
        plt.plot(target_test_xorig, pred_yorig[i], 'b-', linewidth=1.5, label = label_smooth,\
                 alpha = 1/(len(pred_yorig)))
        if mu_target is not None and sigma_target is not None:
            plt.errorbar(target_xorig,pred_y_observed[i],[predicted_var_observed_top[i]-pred_y_observed[i],pred_y_observed[i]-predicted_var_observed_bot[i]],label = label_target, linestyle='', elinewidth=1.3, color='b',alpha = 0.6,marker = 'o')
        #Fill in the different sigma bands
        plt.fill_between(
      target_test_xorig,
      vartop_1[i],
      varbot_1[i],
      alpha=0.6/len(pred_yorig),
      facecolor='#ff9999', label=label1sig,
      interpolate=True)
        plt.fill_between(
      target_test_xorig,
      vartop_2[i],
      varbot_2[i],
      alpha=0.4/len(pred_yorig),
      facecolor='#ff9999', label=label2sig,
      interpolate=True)
        plt.fill_between(
      target_test_xorig,
      vartop_3[i],
      varbot_3[i],
      alpha=0.2/len(pred_yorig),
      facecolor='#ff9999', label=label3sig,
      interpolate=True)
    
    #plt.errorbar(target_xorig, target_yorig, yerr=yerr1[0], linestyle='', elinewidth=1.3, color='k',label = 'observations',markersize=5,marker = 'o')
    plt.errorbar(target_xorig, target_yorig, yerr=0.000001, linestyle='', elinewidth=1.3, color='k',label = 'observations',markersize=5,marker = 'o')

    plt.legend(fontsize=10)
    minx=math.ceil(target_test_xorig.min())
    maxx=math.ceil(target_test_xorig.max())
    middlex=math.ceil((minx+maxx)/2.)
    miny=round( target_yorig.min()-yerr1.numpy().max(),1)
    maxy=round( target_yorig.max()+yerr1.numpy().max(),1)
    middley=round((miny+maxy)/2.)

    # Make the plot pretty
    plt.yticks([miny, middley, maxy])
    plt.xticks([minx, middlex, maxx])

    plt.tick_params(axis='both', which='minor', labelsize=8.5)
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.title(lcName, fontsize=8)

    if cut_little:
        plt.xlim(53700,54200)

    if save:
        if isTrainData and flagval == 0:
            savePath = os.path.join(OUTPUT_PATH, 'train')
        elif notTrainData and flagval == 1:
            savePath = os.path.join(OUTPUT_PATH, 'val')
        else:
            savePath = os.path.join(OUTPUT_PATH, 'test')

        lcName = lcName.split(',')[0]
        if cut_little:
            pltPath = os.path.join(savePath, 'plots', lcName + '_cut.png')
        else:
            pltPath = os.path.join(savePath, 'plots', lcName + '.png')

        if not os.path.exists(os.path.join(savePath, 'plots')):
            os.makedirs(os.path.join(savePath, 'plots'))

        if not os.path.exists(os.path.join(savePath, 'data')):
            os.makedirs(os.path.join(savePath, 'data'))

        plt.savefig(pltPath)
        plt.clf()
        
        for i in range(len(pred_yorig)):
            csvpath = os.path.join(savePath, 'data', lcName + '_predictions_sample_{}.csv'.format(i))
            d = {'mjd': target_test_xorig, 
                'mag': pred_yorig[i],
                'magerr_top_2sig': vartop_2[i]-pred_yorig[i],
                'magerr_bot_2sig':pred_yorig[i]-varbot_2[i]}
            df = pd.DataFrame(data=d)
            df.to_csv(csvpath, index=False)
        
        if os.path.exists(pltPath + ".png") == False:
            #print(pltPath)
            pass
    else:
        print('Not saving')

def load_test_data(DATA_PATH_TEST,num_target_smooth = 400,tf_dir = None,param_df = None, param_columns = None, class_labels_df = None):
    testSet = LightCurvesDataset(root_dir = DATA_PATH_TEST, status = 'test',num_target_smooth = num_target_smooth,tf_dir = tf_dir,param_df = param_df, param_columns = param_columns, class_labels_df = class_labels_df)
    testLoader = DataLoader(testSet,
                             num_workers = 0,
                             batch_size  = 1,      # must remain 1
                             shuffle=True,
                             pin_memory  = True)
    
    return testLoader

def load_train_data(data_path,num_target_smooth = 400,tf_dir = None,param_df = None, param_columns = None, class_labels_df = None):
    train_set = LightCurvesDataset(root_dir=data_path, status='test',num_target_smooth = num_target_smooth,tf_dir = tf_dir,param_df = param_df, param_columns = param_columns, class_labels_df = class_labels_df)
    train_loader = DataLoader(train_set, 
                              num_workers=0, 
                              batch_size=1, 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader

def load_val_data(data_path,num_target_smooth = 400,tf_dir = None,param_df = None, param_columns = None, class_labels_df = None):
    valSet = LightCurvesDataset(root_dir = data_path, status = 'test',num_target_smooth = num_target_smooth,tf_dir = tf_dir,param_df = param_df, param_columns = param_columns, class_labels_df = class_labels_df)
    valLoader = DataLoader(valSet,
                           num_workers = 0,
                           batch_size  = 1, 
                           pin_memory  = True)
    return valLoader

def find_LC_transform(lists, search):
  search1=search
  return list(filter(lambda x:x[0]==search1,lists))

import torch
from tqdm import tqdm

def plot_test_data(model, testLoader, criterion, mseMetric, plot_function2,\
                   device,tr, OUTPUT_PATH = './output/predictions',beta_param = 0, beta_tf = 0, beta_classifier = 0):
    testMetrics = {}
    
    with torch.no_grad():
        zs = []
        Rs = []
        combined_rep = []
        transfer_functions_all = []
        parameters_all = []
        classes_all = []
        
        criterion_tf = LogProbLoss()
        criterion_classifier = nn.CrossEntropyLoss()
        criterion_param = LogProbLoss()
        
        for data in tqdm(testLoader):
            # Unpack data
            lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], \
                                                                              data['target_y'], data['target_test_x'], data['measurement_error']
            
            tf,param,class_labels = data['transfer_function'],data['parameters'],data['tf_class_labels']

            # Move to gpu
            context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                      target_x.to(device), target_y.to(device), \
                                                                      target_test_x.to(device), measurement_error.to(device)
            
            if param[0] != 'None':
                param = param.to(device)
            if tf[0] != 'None':
                tf[0]= tf.to(device)
            if class_labels[0] != 'None':
                class_labels = class_labels.to(device)

            # Forward pass
            dist, mu, sigma,_,_2,_,_,_,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_x)
            
            # Calculate loss
            #loss = criterion(dist, target_y)
            loss_mag = criterion(dist, target_y)
            if predicted_parameters is not None:
                loss_param = criterion_param(predicted_parameters,param)
            else:
                loss_param = 0
            if predicted_tf is not None:
                loss_tf = criterion_tf(predicted_tf,tf)
            else:
                loss_tf = 0
            if predicted_classes is not None:
                loss_classes = criterion_classifier(predicted_classes,class_labels)
            else:
                loss_classes = 0
            loss = loss_mag + beta_param*loss_param + beta_tf*loss_tf + beta_classifier*loss_classes
            
            loss = loss.item()
            
            if len(mu.shape) == 3:
                mu_mse_metric = mu.mean(0)
            else:
                mu_mse_metric = mu

            # Calculate MSE metric
            mseLoss = mseMetric(target_y, mu_mse_metric, measurement_error)

            # Discard .csv part of LC name
            lcName = lcName[0].split('.')[0]

            #Take name for finding coefficinetns for backward propagation 
            llc = lcName                                                                 

            # Add metrics to map
            testMetrics[lcName] = {'log_prob:': str(loss_mag),
                                      'mse': str(mseLoss),
                                     'parametric_loss': str(loss_param),
                                     'transfer_function_loss':str(loss_tf),
                                     'transfer_function_classifier_loss':str(loss_classes),  
                                   'total_loss':str(loss)
                                  }

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{-loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))
            
            #Copy the targets to plot
            mu_target = mu.detach().clone()
            sigma_target = sigma.detach().clone()
            
            # Predict and plot
            dist, mu, sigma,_,z,R,latent_dist,agg_R_z,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_test_x)

            #coeeficinets for transformation back
            ZX = find_LC_transform(tr, llc[:7])
            
            plot_function2(tr, target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, OUTPUT_PATH, save=True, isTrainData=False, flagval=0, mu_target = mu_target, sigma_target = sigma_target,cut_little = False)


            if z is not None:
                zs.append(z.detach().numpy())
            else:
                zs.append(z)
            
            if R is not None:
                Rs.append(R.detach().numpy())
            else:
                Rs.append(R)
            
            if agg_R_z is not None:
                combined_rep.append(agg_R_z.detach())
            else:
                combined_rep.append(agg_R_z)
                
            parameters_all.append(predicted_parameters)
                
            transfer_functions_all.append(predicted_tf)
                
            if predicted_classes is not None:
                classes_all.append(predicted_classes.detach().numpy())
            else:
                classes_all.append(predicted_classes)
                

    return testMetrics,np.array(zs),Rs,combined_rep, np.array(parameters_all),np.array(transfer_functions_all),np.array(classes_all)


def save_test_metrics(OUTPUT_PATH, testMetrics):
    with open(OUTPUT_PATH + 'test/testMetrics.json', 'w') as fp:
        json.dump(testMetrics, fp, indent=4)

        
def plot_train_light_curves(model, trainLoader, criterion, mseMetric, plot_function2,\
                   device,tr, OUTPUT_PATH = './output/predictions',beta_param = 0, beta_tf = 0, beta_classifier = 0):
    """
    Plots light curves from test set in original range MJD x Mag

    Args:
        model (torch.nn.Module): Trained probabilistic model
        criterion (torch.nn.Module): Loss function
        mseMetric (function): Mean squared error metric
        trainLoader (torch.utils.data.DataLoader): Dataloader for training set
        device (str): Device for PyTorch model
        tr (pd.DataFrame): Transforming coefficients DataFrame for LCs
        CI(float): The Confidence Interval to plot

    Returns:
        trainMetrics (dict): Log probability and MSE metrics for train set

    """
    trainMetrics = {}
    
    with torch.no_grad():
        zs = []
        Rs = []
        combined_rep = []
        transfer_functions_all = []
        parameters_all = []
        classes_all = []
        
        criterion_tf = LogProbLoss()
        criterion_classifier = nn.CrossEntropyLoss()
        criterion_param = LogProbLoss()
        
        for data in tqdm(trainLoader):
            # Unpack data
            lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], \
                                                                              data['target_y'], data['target_test_x'], data['measurement_error']
            
            tf,param,class_labels = data['transfer_function'],data['parameters'],data['tf_class_labels']

            # Move to gpu
            context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                      target_x.to(device), target_y.to(device), \
                                                                      target_test_x.to(device), measurement_error.to(device)
            
            if param[0] != 'None':
                param = param.to(device)
            if tf[0] != 'None':
                tf = tf.to(device)
            if class_labels[0] != 'None':
                class_labels = class_labels.to(device)

            # Forward pass
            dist, mu, sigma,_,_2,_,_,_,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_x)
            
            # Calculate loss
            #loss = criterion(dist, target_y)
            loss_mag = criterion(dist, target_y)
            if predicted_parameters is not None:
                loss_param = criterion_param(predicted_parameters,param)
            else:
                loss_param = 0
            if predicted_tf is not None:
                loss_tf = criterion_tf(predicted_tf,tf)
            else:
                loss_tf = 0
            if predicted_classes is not None:
                loss_classes = criterion_classifier(predicted_classes.mean(dim=0),class_labels)
            else:
                loss_classes = 0
            loss = loss_mag + beta_param*loss_param + beta_tf*loss_tf + beta_classifier*loss_classes
            
            loss = loss.item()
            
            if len(mu.shape) == 3:
                mu_mse_metric = mu.mean(0)
            else:
                mu_mse_metric = mu

            # Calculate MSE metric
            mseLoss = mseMetric(target_y, mu_mse_metric, measurement_error)

            # Discard .csv part of LC name
            lcName = lcName[0].split('.')[0]

            #Take name for finding coefficinetns for backward propagation 
            llc = lcName                                                                 

            # Add metrics to map
            trainMetrics[lcName] = {'log_prob:': str(loss_mag),
                                      'mse': str(mseLoss),
                                     'parametric_loss': str(loss_param),
                                     'transfer_function_loss':str(loss_tf),
                                     'transfer_function_classifier_loss':str(loss_classes),  
                                   'total_loss':str(loss)
                                  }

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{-loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))
            
            #Copy the targets to plot
            mu_target = mu.detach().clone()
            sigma_target = sigma.detach().clone()
            
            # Predict and plot
            dist, mu, sigma,_,z,R,latent_dist,agg_R_z,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_test_x)

            #coeeficinets for transformation back
            ZX = find_LC_transform(tr, llc[:7])

            plot_function2(tr, target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, OUTPUT_PATH, save=True, isTrainData=True, flagval=0, mu_target = mu_target, sigma_target = sigma_target,cut_little = False)

            if z is not None:
                zs.append(z.detach().numpy())
            else:
                zs.append(z)
            
            if R is not None:
                Rs.append(R.detach().numpy())
            else:
                Rs.append(R)
            
            if agg_R_z is not None:
                combined_rep.append(agg_R_z.detach())
            else:
                combined_rep.append(agg_R_z)
                
            parameters_all.append(predicted_parameters)
                
            transfer_functions_all.append(predicted_tf)
                
            if predicted_classes is not None:
                classes_all.append(predicted_classes.detach().numpy())
            else:
                classes_all.append(predicted_classes)
                

    return trainMetrics,np.array(zs),Rs,combined_rep, np.array(parameters_all),np.array(transfer_functions_all),np.array(classes_all)

def save_train_metrics(OUTPUT_PATH, trainMetrics):
    with open(OUTPUT_PATH + 'train/trainMetrics.json', 'w') as fp:
        json.dump(trainMetrics, fp, indent=4)

import torch
from tqdm import tqdm


def plot_val_curves(model, valLoader, criterion, mseMetric, plot_function2,\
                   device,tr, OUTPUT_PATH = './output/predictions',beta_param = 0, beta_tf = 0, beta_classifier = 0):
    """
    Plots light curves from val set in original range MJD x Mag

    Args:
        model (torch.nn.Module): Trained probabilistic model
        criterion (torch.nn.Module): Loss function
        mseMetric (function): Mean squared error metric
        valLoader (torch.utils.data.DataLoader): Dataloader for validation set
        device (str): Device for PyTorch model
        tr (pd.DataFrame): Transforming coefficients DataFrame for LCs
        CI(float): The Confidence Interval to plot

    Returns:
        trainMetrics (dict): Log probability and MSE metrics for train set

    """
    valMetrics = {}
    
    with torch.no_grad():
        zs = []
        Rs = []
        combined_rep = []
        transfer_functions_all = []
        parameters_all = []
        classes_all = []
        
        criterion_tf = LogProbLoss()
        criterion_classifier = nn.CrossEntropyLoss()
        criterion_param = LogProbLoss()
        
        for data in tqdm(valLoader):
            # Unpack data
            lcName, context_x, context_y, target_x, target_y, target_test_x, measurement_error = data['lcName'], data['context_x'], \
                                                                              data['context_y'], data['target_x'], \
                                                                              data['target_y'], data['target_test_x'], data['measurement_error']
            
            tf,param,class_labels = data['transfer_function'],data['parameters'],data['tf_class_labels']

            # Move to gpu
            context_x, context_y, target_x, target_y, target_test_x, measurement_error = context_x.to(device), context_y.to(device), \
                                                                      target_x.to(device), target_y.to(device), \
                                                                      target_test_x.to(device), measurement_error.to(device)
            
            if param[0] != 'None':
                param = param.to(device)
            if tf[0] != 'None':
                tf = tf.to(device)
            if class_labels[0] != 'None':
                class_labels = class_labels.to(device)

            # Forward pass
            dist, mu, sigma,_,_2,_,_,_,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_x)
            
            # Calculate loss
            #loss = criterion(dist, target_y)
            loss_mag = criterion(dist, target_y)
            if predicted_parameters is not None:
                loss_param = criterion_param(predicted_parameters,param)
            else:
                loss_param = 0
            if predicted_tf is not None:
                loss_tf = criterion_tf(predicted_tf,tf)
            else:
                loss_tf = 0
            if predicted_classes is not None:
                loss_classes = criterion_classifier(predicted_classes.mean(dim=0),class_labels)
            else:
                loss_classes = 0
            loss = loss_mag + beta_param*loss_param + beta_tf*loss_tf + beta_classifier*loss_classes
            
            loss = loss.item()
            
            if len(mu.shape) == 3:
                mu_mse_metric = mu.mean(0)
            else:
                mu_mse_metric = mu

            # Calculate MSE metric
            mseLoss = mseMetric(target_y, mu_mse_metric, measurement_error)

            # Discard .csv part of LC name
            lcName = lcName[0].split('.')[0]

            #Take name for finding coefficinetns for backward propagation 
            llc = lcName                                                                 

            # Add metrics to map
            valMetrics[lcName] = {'log_prob:': str(loss_mag),
                                      'mse': str(mseLoss),
                                     'parametric_loss': str(loss_param),
                                     'transfer_function_loss':str(loss_tf),
                                     'transfer_function_classifier_loss':str(loss_classes),  
                                   'total_loss':str(loss)
                                  }

            # Add loss value to LC name
            lcName = lcName + ", loss: " + str(float(f'{-loss:.2f}')) + ", MSE: " + str(float(f'{mseLoss:.2f}'))
            
            #Copy the targets to plot
            mu_target = mu.detach().clone()
            sigma_target = sigma.detach().clone()
            
            # Predict and plot
            dist, mu, sigma,_,z,R,latent_dist,agg_R_z,predicted_parameters, predicted_tf, predicted_classes = model(context_x, context_y, target_test_x)

            #coeeficinets for transformation back
            ZX = find_LC_transform(tr, llc[:7])

            plot_function2(tr, target_x, target_y, context_x, context_y, measurement_error, mu, sigma, target_test_x, lcName, OUTPUT_PATH, save=True, isTrainData=False, notTrainData = True,flagval=1, mu_target = mu_target, sigma_target = sigma_target)
            
            if z is not None:
                zs.append(z.detach().numpy())
            else:
                zs.append(z)
            
            if R is not None:
                Rs.append(R.detach().numpy())
            else:
                Rs.append(R)
            
            if agg_R_z is not None:
                combined_rep.append(agg_R_z.detach())
            else:
                combined_rep.append(agg_R_z)
                
            parameters_all.append(predicted_parameters)
                
            transfer_functions_all.append(predicted_tf)
                
            if predicted_classes is not None:
                classes_all.append(predicted_classes.detach().numpy())
            else:
                classes_all.append(predicted_classes)
                

    return valMetrics,np.array(zs),Rs,combined_rep, np.array(parameters_all),np.array(transfer_functions_all),np.array(classes_all)

def save_val_metrics(OUTPUT_PATH, valMetrics):
    with open(OUTPUT_PATH + 'val/valMetrics.json', 'w') as fp:
        json.dump(valMetrics, fp, indent=4)

def Plotting_TF_Mean(predicted_tf,actual_tf,ttau,TF_SAVE_PATH,ttau_plot_len = -1):
    #Plotting the mean transfer function
    #Get the length of the predictions
    n = len(predicted_tf)
    # Calculate the average of the means
    mean_avg = sum(dist.loc for dist in predicted_tf) / n
    # Calculate the average of the variances
    variance_avg = sum(dist.scale**2 for dist in predicted_tf) / (n**2)
    # The standard deviation is the square root of the averaged variance
    std_avg = torch.sqrt(variance_avg)
    # Create the resulting normal distribution
    averaged_tf = Normal(loc=mean_avg, scale=std_avg)

    #PLot the mean transfer function of the predicted and actual tfs
    plt.figure()
    plt.plot(ttau[:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len],label = 'Prediction',color='b')
    plt.fill_between(ttau[:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]+averaged_tf.scale.numpy()[0][:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]-averaged_tf.scale.numpy()[0][:ttau_plot_len],alpha = 0.6,label = r'$1\sigma$ CI',color = '#ff9999')
    plt.fill_between(ttau[:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]+2*averaged_tf.scale.numpy()[0][:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]-2*averaged_tf.scale.numpy()[0][:ttau_plot_len],alpha = 0.4,label = r'$2\sigma$ CI',color = '#ff9999')
    plt.fill_between(ttau[:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]+3*averaged_tf.scale.numpy()[0][:ttau_plot_len],averaged_tf.loc.numpy()[0][:ttau_plot_len]-3*averaged_tf.scale.numpy()[0][:ttau_plot_len],alpha = 0.2,label = r'$3\sigma$ CI',color = '#ff9999')
    plt.plot(ttau[:ttau_plot_len],np.mean(actual_tf,axis = 0)[:ttau_plot_len],label = 'Real',color='k')
    plt.xlabel(r'$\tau$ (light days)')
    plt.ylabel('Transfer Function')
    plt.title(f'Mean Transfer Function - LogProbLoss: {-tf_loss(averaged_tf,torch.tensor(np.mean(actual_tf,axis = 0))):.2f}')
    plt.legend()
    plt.savefig(TF_SAVE_PATH+'Mean.png')
    plt.close()    

def Plotting_TF_Individual(dataLoader,predicted_tf,actual_tf,ttau,TF_SAVE_PATH,ttau_plot_len = -1):
    #Plot each transfer function individually
    for num in tqdm(range(len(dataLoader))):
        this_name = dataLoader.dataset[num]['lcName']
        loss_one_function = tf_loss(predicted_tf[num],dataLoader.dataset[num]['transfer_function'])
        plt.title(f'{this_name}, LogProbLoss: {-loss_one_function:.2f}')
        mean_predicted = predicted_tf[num].loc.numpy()[0]
        std_predicted = predicted_tf[num].scale.numpy()[0]
        plt.plot(ttau[:ttau_plot_len],mean_predicted[:ttau_plot_len],label = 'Prediction',color='b')
        plt.fill_between(ttau[:ttau_plot_len],mean_predicted[:ttau_plot_len]+std_predicted[:ttau_plot_len],mean_predicted[:ttau_plot_len]-std_predicted[:ttau_plot_len],alpha = 0.6,label = r'$1\sigma$ CI',color = '#ff9999')
        plt.fill_between(ttau[:ttau_plot_len],mean_predicted[:ttau_plot_len]+2*std_predicted[:ttau_plot_len],mean_predicted[:ttau_plot_len]-std_predicted[:ttau_plot_len],alpha = 0.4,label = r'$2\sigma$ CI',color = '#ff9999')
        plt.fill_between(ttau[:ttau_plot_len],mean_predicted[:ttau_plot_len]+3*std_predicted[:ttau_plot_len],mean_predicted[:ttau_plot_len]-std_predicted[:ttau_plot_len],alpha = 0.2,label = r'$3\sigma$ CI',color = '#ff9999')
        plt.plot(ttau[:ttau_plot_len],actual_tf[num][:ttau_plot_len],label = 'Actual',color='k')
        plt.xlabel(r'$\tau$ (light days)')
        plt.ylabel('Transfer Function')
        plt.legend()
        plt.savefig(TF_SAVE_PATH+f'{this_name}.png')
        plt.close()

def Plotting_Param_Individual(dataLoader,predicted_params,actual_params,columns,PARAM_SAVE_PATH):
    for i,column_name in enumerate(columns):
        plt.figure()
        mean_predicted_parameters = []
        actual_parameters = []
        std_predicted_parameters = []
        for num in tqdm(range(len(dataLoader))):
            this_name = dataLoader.dataset[num]['lcName']
            mean_predicted = predicted_params[num].loc.numpy()[0][i]
            mean_predicted_parameters.append(mean_predicted)
            std_predicted = predicted_params[num].scale.numpy()[0][i]
            std_predicted_parameters.append(std_predicted)
            actual_param = actual_params[num][i]
            actual_parameters.append(actual_param)
        plt.scatter(actual_parameters,mean_predicted_parameters,color = 'b')
        plt.errorbar(actual_parameters,mean_predicted_parameters,yerr = std_predicted_parameters,linestyle = '',color = '#ff9999',alpha = 0.8,label = r'$1\sigma$ CI')
        plt.errorbar(actual_parameters,mean_predicted_parameters,yerr = 2*np.array(std_predicted_parameters),linestyle = '',color = '#ff9998',alpha = 0.6,label = r'$2\sigma$ CI')
        plt.errorbar(actual_parameters,mean_predicted_parameters,yerr = 3*np.array(std_predicted_parameters),linestyle = '',color = '#ff9997',alpha = 0.4,label = r'$3\sigma$ CI')
        plt.title(column_name)
        plt.plot(np.linspace(min(actual_parameters),max(actual_parameters),1000),np.linspace(min(actual_parameters),max(actual_parameters),1000),linestyle = ':',label = '1:1',color = 'k')
        plt.xlabel('Actual Parameter')
        plt.ylabel('Predicted Parameter')
        plt.legend()
        plt.savefig(PARAM_SAVE_PATH+f'{column_name}.png')