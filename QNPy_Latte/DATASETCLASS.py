import random
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch

class LightCurvesDataset(Dataset):
    """Dataset class."""
    
    def __init__(self, root_dir, status,num_target_smooth = 400, tf_dir = None, param_df = None, param_columns = None, class_labels_df = None):
        self.root_dir = root_dir
        self.status = status
        self.num_target_smooth = num_target_smooth
        self.file_paths = []
        self.tf_paths = []
        self.param = param_df
        self.param_columns = param_columns
        self.class_labels = class_labels_df
        self.tf_dir = tf_dir
        for file_name in os.listdir(self.root_dir):
            self.file_paths.append(os.path.join(self.root_dir, file_name))
            if tf_dir is not None:
                tf_file_name = file_name.split('_')[0]+'.csv'
                self.tf_paths.append(os.path.join(self.tf_dir,tf_file_name))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Reads one light curve data and picks target and context points."""
        
        # Read data
        lcName = os.path.basename(self.file_paths[idx])
        data = pd.read_csv(self.file_paths[idx])
        x_data = data['time']
        y_data = data['cont'] 
        z_data = data['conterr']  ### input error

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        num_total = np.size(x_data)
        
        context_x = np.copy(x_data)
        context_y = np.copy(y_data)

        target_x = np.copy(x_data)
        target_y = np.copy(y_data)
        
        measurement_error = np.copy(z_data)
        
        if self.status == 'test':
            # Select targets on the whole curve
            target_test_x = np.linspace(-2., 2., num = self.num_target_smooth)
            
        # Convert to tensors
        target_x = torch.Tensor(target_x)
        target_y = torch.Tensor(target_y)
        context_x = torch.Tensor(context_x)
        context_y = torch.Tensor(context_y)
        measurement_error = torch.Tensor(measurement_error)
            
        
        # Squeeze first dimension
        target_x = torch.squeeze(target_x, 0)
        target_y = torch.squeeze(target_y, 0)
        context_x = torch.squeeze(context_x, 0)
        context_y = torch.squeeze(context_y, 0)
        measurement_error = torch.squeeze(measurement_error, 0)
        
        # Get the transfer function and parameters if possible
        if self.tf_dir is not None:
            tf = pd.read_csv(self.tf_paths[idx])
            tf = torch.Tensor(tf.tf/max(tf.tf))
            tf = torch.squeeze(tf,0)
        else:
            tf = 'None'
        if self.param is not None:
            param = self.param[self.param.Label == int(lcName.split('_')[0])][self.param_columns].to_numpy()
            param = torch.Tensor(param)
            param = torch.squeeze(param,0)
        else:
            param = 'None'
        if self.class_labels is not None:
            class_labels = self.class_labels[self.class_labels.Name == int(lcName.split('_')[0])].tf_label.to_numpy()
            class_labels = torch.tensor(class_labels,dtype = torch.long)
            class_labels = torch.squeeze(class_labels,0)
        else:
            class_labels = 'None'
      
        if self.status == 'train': 
            data = {'lcName' : lcName,
                    'context_x': context_x,
                    'context_y': context_y,
                    'target_x': target_x,
                    'target_y': target_y,
                    'measurement_error': measurement_error,
                   'transfer_function': tf,
                   'parameters':param,
                   'tf_class_labels':class_labels}
        else:
            target_test_x = torch.Tensor(target_test_x)
            target_test_x = torch.squeeze(target_test_x)

            data = {'lcName' : lcName,
                    'context_x': context_x,
                    'context_y': context_y,
                    'target_x': target_x,
                    'target_y': target_y,
                    'measurement_error': measurement_error,
                    'target_test_x': target_test_x,
                   'transfer_function': tf,
                   'parameters':param,
                   'tf_class_labels':class_labels}

        return data
    
def collate_lcs(batch,augment = True):
    """Custom collate function for padding and stacking tensors in a batch.
    
    Args:
          batch: List containing variable length tensors where each item represents data for one light curve
                     data = {'lcName' : lcName,
                             'context_x': context_x,
                             'context_y': context_y,
                             'target_x': target_x,
                             'target_y': target_y,
                             'measurement_error': z_data}
        Returns:
          [context_x, context_y, target_x], target_y: Padded and stacked tensors.
    """
    
    # Calculate max num_total points
    num_total = None
    for item in batch:
        if num_total is None:
            num_total = item['context_x'].shape[0]
        else:
            num_total = min(num_total, item['context_x'].shape[0])
        
    # Determine number of context points
    upper_bound = int(num_total * 80/100)      # 80% of total points
    lower_bound = int(num_total * 60/100)      # 60% of total points
    num_context = random.randint(lower_bound, upper_bound)
    
    # Determine number of target points
    num_target = random.randint(num_context, num_total)
    
    context_x = []
    context_y = []
    target_x  = []
    target_y  = []
    measurement_error = []
    tfs = []
    params = []
    all_labels = []
    for item in batch:
        # Calculate the target indices
        start_index_target = random.randint(0, int(num_total * (1.0 - num_target/num_total)))
        target_indices = list(range(start_index_target, start_index_target + num_target))
        target_indices = [idx % num_total for idx in target_indices]
        
        # Calculate the context indices
        start_index_context = random.randint(start_index_target, start_index_target + (num_target-num_context))
        context_indices = list(range(start_index_context, start_index_context + num_context))
        context_indices = [idx % num_total for idx in context_indices]
        
        
        # Pad and append to list
        if augment:
            context_y_one_item = (
            item['context_y'][context_indices]
            + torch.randn_like(item['measurement_error'][context_indices])
            * item['measurement_error'][context_indices])

            target_y_one_item = (
            item['target_y'][target_indices]
            + torch.randn_like(item['measurement_error'][target_indices])
            * item['measurement_error'][target_indices]
    )
        else:
            context_y_one_item = item['context_y'][context_indices]
            target_y_one_item = item['target_y'][target_indices]

        context_x.append(item['context_x'][context_indices]) 
        context_y.append(context_y_one_item) 
        target_x.append(item['target_x'][target_indices]) 
        target_y.append(target_y_one_item) 
        measurement_error.append(item['measurement_error'][target_indices]) 
        tfs.append(item['transfer_function'])
        params.append(item['parameters'])
        all_labels.append(item['tf_class_labels'])
    
    # Stack tensors
    context_x = torch.stack(context_x)
    context_y = torch.stack(context_y)
    target_x  = torch.stack(target_x)
    target_y  = torch.stack(target_y)
    measurement_error = torch.stack(measurement_error)
    if tfs[0] != 'None':
        tfs = torch.stack(tfs)
    else:
        tfs = 'None'
    if params[0] != 'None':
        params = torch.stack(params)
    else:
        params = 'None'
    if all_labels[0] != 'None':
        all_labels = torch.stack(all_labels)
    else:
        all_labels = 'None'
    
    return [context_x, context_y, target_x, measurement_error], target_y, tfs,params,all_labels