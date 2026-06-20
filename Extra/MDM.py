import joblib

import PREDICTION as por #Importing PREDICTION_onePDF module from the package
from PREDICTION import * #Importing importing all packages from PREDICTION_onePDF module
#The functions plot_function2, back_x and back_y must be imported separately
from PREDICTION import plot_function2, back_x, back_y, find_LC_transform 
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.optim as optim
from torch.distributions import Normal
import seaborn as sns

class RollingWindow(nn.Module):
    def __init__(self, window_size):
        super(RollingWindow, self).__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=window_size,
            stride=1,
            padding = 'same',
            bias=False
        )
        # Initialize kernel to be uniform for averaging (optional)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / window_size)

    def forward(self, x):
        # x: (batch_size, num_time_steps)
        x = x.unsqueeze(1)  # Add channel dimension -> (batch_size, 1, num_time_steps)
        rolling_output = self.conv(x)  # (batch_size, 1, num_time_steps)
        return rolling_output.squeeze(1)  # Remove channel dimension -> (batch_size, num_time_steps


class MixtureDensityNetwork_Param(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_components=100):
        super(MixtureDensityNetwork_Param, self).__init__()
        full_output_size = num_components*output_size
        hidden_size1 = 128
        #layer_size4 = int((hidden_size+7*full_output_size)/8)
        self.num_components = num_components
        self.hidden_layer = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.output_size = output_size
        self.rolling_window = RollingWindow(100)

        # Each Gaussian component has a mean, a log standard deviation, and a weight
        self.mean_head = nn.Linear(hidden_size1, full_output_size)
        self.log_std_head = nn.Linear(hidden_size1, full_output_size)
        self.weight_head = nn.Linear(hidden_size1, num_components)
        
    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        hidden = self.dropout(hidden)
        
        # Compute parameters for each component
        means = self.mean_head(hidden).view(-1, self.num_components, self.output_size)  # Shape: [batch_size, num_components, output_size]
        log_stds = self.log_std_head(hidden).view(-1, self.num_components, self.output_size)  # Shape: [batch_size, num_components, output_size]
        #stds = 0.01 + 0.99 * torch.exp(log_stds)  # Ensures positive std
        stds = F.elu(log_stds)+1+1e-15  # Ensures positive std

        # Compute component weights and apply softmax to get probabilities
        #weights = F.gumbel_softmax(self.weight_head(hidden), dim=-1,tau = 1.03) # Shape: [batch_size, num_components]
        weights = F.softmax(self.weight_head(hidden), dim=-1) # Shape: [batch_size, num_components]


        return [means, stds, weights]

class MixtureDensityNetwork_TF(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_components=100):
        super(MixtureDensityNetwork_TF, self).__init__()
        full_output_size = num_components*output_size
        hidden_size = int((input_size+full_output_size)/2)
        hidden_size1 = int((input_size+full_output_size)/2)
        #layer_size4 = int((hidden_size+7*full_output_size)/8)
        self.num_components = num_components
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()
        self.output_size = output_size
        self.rolling_window = RollingWindow(30)

        # Each Gaussian component has a mean, a log standard deviation, and a weight
        self.mean_head = nn.Linear(hidden_size, full_output_size)
        self.log_std_head = nn.Linear(hidden_size, full_output_size)
        self.weight_head = nn.Linear(hidden_size, num_components)
        
    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        hidden = self.dropout(hidden)

        hidden = self.rolling_window(hidden)

        # Compute parameters for each component
        means = self.mean_head(hidden).view(-1, self.num_components, self.output_size)  # Shape: [batch_size, num_components, output_size]
        log_stds = self.log_std_head(hidden).view(-1, self.num_components, self.output_size)  # Shape: [batch_size, num_components, output_size]
        #stds = 0.01 + 0.99 * torch.exp(log_stds)  # Ensures positive std
        stds = F.elu(log_stds)+1+1e-15  # Ensures positive std

        # Compute component weights and apply softmax to get probabilities
        #weights = F.gumbel_softmax(self.weight_head(hidden), dim=-1,tau = 1.0) # Shape: [batch_size, num_components]
        weights = F.softmax(self.weight_head(hidden), dim=-1) # Shape: [batch_size, num_components]

        return [means, stds, weights]

class MixtureLogProbLoss(nn.Module):
    def __init__(self):
        super(MixtureLogProbLoss, self).__init__()

    def forward(self,dist, y_true):
        # Create the Gaussian distributions for each component
        means = dist[0] 
        stds = dist[1]
        weights = dist[2]
        
        gaussians = Normal(means, stds)
        y_true = y_true.unsqueeze(1)  # Shape: [batch_size, 1, output_size] for broadcasting

        # Compute log-probabilities for each component
        log_probs = gaussians.log_prob(y_true)  # Shape: [batch_size, num_components, output_size]
        log_probs = log_probs.mean(-1)  # Sum over output dimensions to get scalar log-prob for each component

        # Convert component log-probs to mixture log-prob using weights
        weighted_log_probs = log_probs + torch.log(weights)  # Shape: [batch_size, num_components]
        log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)  # Aggregate over components

        # Negative log likelihood
        loss = -log_sum_exp.mean()
        return loss

def sample_from_mdn(mdn_output, num_samples=10,scaler = None):
    means, stds, weights = mdn_output  # Shapes: means, stds -> [batch_size, num_components, output_size]; weights -> [batch_size, num_components]
    
    batch_size, num_components, output_size = means.shape
    
    # Draw samples from each component and weight them
    all_samples = []
    for _ in range(num_samples):
        # Sample component indices based on weights for each batch item
        component_indices = torch.multinomial(weights, num_samples=1, replacement=True).squeeze(-1)  # Shape: [batch_size]
        
        # Expand component_indices to match means and stds shape for gathering
        component_indices = component_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, output_size)  # Shape: [batch_size, 1, output_size]

        # Gather means and stds for the chosen components across the output dimensions
        chosen_means = torch.gather(means, 1, component_indices).squeeze(1)  # Shape: [batch_size, output_size]
        chosen_stds = torch.gather(stds, 1, component_indices).squeeze(1)    # Shape: [batch_size, output_size]
        
        # Sample from the chosen Gaussian components
        samples = chosen_means + chosen_stds * torch.randn_like(chosen_means)  # Shape: [batch_size, output_size]
        if scaler:
            all_samples.append(torch.from_numpy(scaler.inverse_transform(samples)))
        else:
            all_samples.append(samples)
    
    # Stack all samples and take the mean across them to reduce variance if needed
    final_samples = torch.stack(all_samples, dim=0)  # Shape: [num_samples, batch_size, output_size]
    return final_samples

def train_and_evaluate(z_arrays, target_quantity_arrays, ttau = np.linspace(0.001,8,1000),num_epochs=100, learning_rate=0.01,transfer_function = False,plot = False,patience = 5,batch_size = 8,xlabels = None,no_of_bins = 5,scale = False,scaler_type = 'minmax',save_path = './',param_names = 'Mass',band = 'u',num_components = 5,num_samples = 100,num_channels = 5):
    input_size = z_arrays[0].shape[-1]
    output_size = target_quantity_arrays[0].shape[-1]
    #Grid for time lags in light days
    date_range = ttau
    hidden_size = int((input_size+output_size)/2)

    #If needed to scale
    if scale:
        # Initialize scalers
        if scaler_type == 'minmax':
            scaler_z = MinMaxScaler(feature_range =(-2,2))
            scaler_target = MinMaxScaler(feature_range =(-2,2))
        elif scaler_type == 'standard':
            scaler_z = StandardScaler()
            scaler_target = StandardScaler()
        train_quantity = scaler_target.fit_transform(target_quantity_arrays[1])
        test_quantity = scaler_target.transform(target_quantity_arrays[0])
        val_quantity = scaler_target.transform(target_quantity_arrays[2])
    else:
        train_quantity = target_quantity_arrays[1]
        test_quantity = target_quantity_arrays[0]
        val_quantity = target_quantity_arrays[2]

    train_z = z_arrays[1]
    test_z = z_arrays[0]
    val_z = z_arrays[2]

    if transfer_function:
        train_quantity = target_quantity_arrays[1]/target_quantity_arrays[1].max(axis = 1,keepdims = True)
        test_quantity = target_quantity_arrays[0]/target_quantity_arrays[0].max(axis = 1,keepdims = True)
        val_quantity = target_quantity_arrays[2]/target_quantity_arrays[2].max(axis = 1,keepdims = True)
    #The zs
    train_z = torch.tensor(train_z,dtype = torch.float)
    test_z = torch.tensor(test_z,dtype = torch.float)
    val_z = torch.tensor(val_z,dtype = torch.float)

    
    #The target quantities
    train_quantity = torch.tensor(train_quantity,dtype = torch.float)
    test_quantity = torch.tensor(test_quantity,dtype = torch.float)
    val_quantity = torch.tensor(val_quantity,dtype = torch.float)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_z, train_quantity)
    val_dataset = TensorDataset(val_z, val_quantity)
    test_dataset = TensorDataset(test_z, test_quantity)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)
        
    #Initialize Model and criteria
    if transfer_function:
        model = MixtureDensityNetwork_TF(input_size,output_size,hidden_size,num_components=num_components)
        criterion = MixtureLogProbLoss()
    else:
        model = MixtureDensityNetwork_Param(input_size,output_size,hidden_size,num_components=num_components)
        criterion = MixtureLogProbLoss()
    
    #print(criterion)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay= 1e-4)   
    
    best_model = model.state_dict()
    best_val_loss = float('inf')
    patience_counter = 0
    
    training_curve = []
    val_curve = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
        train_loss /= len(train_loader)
        #print(f'Train Loss ({epoch}): {train_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        #print(f'Validation Loss: {val_loss:.4f}')
        training_curve.append(train_loss)
        val_curve.append(val_loss)

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    
    # Evaluate on test set
    os.makedirs(f'{save_path}/{band}/TF',exist_ok=True)
    os.makedirs(f'{save_path}/{band}/Params',exist_ok=True)
    
    # Load the best model
    model.load_state_dict(best_model)
    if transfer_function:
        torch.save(best_model, f'{save_path}/{band}/TF/model_tf.pth')
    else:
        if len(param_names) == 1:
            torch.save(best_model, f'{save_path}/{band}/Params/model_{param_names[0]}.pth')
        else:
            torch.save(best_model, f'{save_path}/{band}/Params/model_params.pth')

    if scale:
        joblib.dump(scaler_target, f'{save_path}/{band}/param_scaler.gz')
        
    model.eval()
    test_loss = 0
    with torch.no_grad():
        count = 0
        outputs_full = []
        targets_full = []
        std_full = []
        names_full = []
        everything = []
        weights = []
        means = []
        stds = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            #if not transfer_function:     
            weights.append(outputs[2])
            means.append(outputs[0])
            stds.append(outputs[1])
            
            # Sample predictions from MDN
            if scale:
                samples = sample_from_mdn(outputs, num_samples=num_samples,scaler = scaler_target)
            else:
                samples = sample_from_mdn(outputs, num_samples=num_samples,scaler = None)

            if transfer_function:
                # Calculate mean and std across the samples for the transfer function plot
                mean_pred = samples.mean(axis=0)[0]
                std_pred = samples.std(axis=0)[0]
                # Set up the plot
                plot_indices = 50
                plt.figure(figsize=(10, 6))
                plt.plot(date_range[:plot_indices], mean_pred[:plot_indices], label='Predicted Mean', color='b')
                plt.plot(date_range[:plot_indices], targets[0][:plot_indices], label='Target', color='k')
                
                # Confidence Intervals
                plt.fill_between(
                    date_range[:plot_indices],
                    mean_pred[:plot_indices] + std_pred[:plot_indices],
                    mean_pred[:plot_indices] - std_pred[:plot_indices],
                    alpha=0.5, color='#ff9999', label=r'1$\sigma$ CI'
                )
                plt.fill_between(
                    date_range[:plot_indices],
                    mean_pred[:plot_indices] + 2 * std_pred[:plot_indices],
                    mean_pred[:plot_indices] - 2 * std_pred[:plot_indices],
                    alpha=0.3, color='#ff9999', label=r'2$\sigma$ CI'
                )
                plt.fill_between(
                    date_range[:plot_indices],
                    mean_pred[:plot_indices] + 3 * std_pred[:plot_indices],
                    mean_pred[:plot_indices] - 3 * std_pred[:plot_indices],
                    alpha=0.1, color='#ff9999', label=r'3$\sigma$ CI'
                )
                
                plt.title(f'Transfer Function Prediction for {names[:len(test_z)][count]}, Loss: {-loss.item():.4f}')
                plt.xlabel(r'$\tau$ (light days)')
                plt.ylabel('Transfer Function')
                plt.legend()
                plt.ylim(bottom=-0.01)
                plt.savefig(f'{save_path}/{band}/TF/{names[:len(test_z)][count]}.png')
                plt.close()
                pd.DataFrame({'Days':date_range,'TF':mean_pred,'TF_err':std_pred}).to_csv(f'{save_path}/{band}/TF/{names[:len(test_z)][count]}.csv')
            else:
                # Handle parameter predictions as before
                outputs_full.append(np.median(samples.squeeze(-2).numpy(), axis=0))
                if scale:
                    targets_full.append(scaler_target.inverse_transform(targets.numpy())[0])
                else:
                    targets_full.append(targets.numpy()[0])
                std_full.append(np.std(samples.squeeze(-2).numpy(), axis=0))
                name_one = names[:len(test_z)][count]
                names_full.append(name_one)
                
                # Create a DataFrame with 'Names' repeated for each sample
                df = pd.DataFrame({'Names': [name_one] * samples.shape[0]})
                
                # Add each parameter to the DataFrame as separate columns
                for i, param_name in enumerate(param_names):
                    df[param_name] = samples.squeeze(-2).numpy()[:, i]
                everything.append(df)
                
            count += 1
        if not transfer_function:
            for i in range(len(param_names)):
                plt.figure()
                plt.errorbar(targets_full[i],outputs_full[i],yerr = std_full[i],label =r'1$\sigma$ CI',linestyle = '',marker = 'o')
                #plt.scatter(targets_full,outputs_full,label =r'1$\sigma$ CI')
                plt.plot(targets_full[i],targets_full[i],label = '1:1',linestyle = ':')
                plt.xlabel('Real Parameters')
                plt.ylabel('Predicted Parameters')
                plt.legend()
                plt.savefig(f'{save_path}/{band}/Params/{param_names[i]}_Scatter.png')
                plt.close()
                plt.figure()
                plt.hist(outputs_full[i],label = 'Predicted',bins = np.linspace(min(outputs_full[i]),max(outputs_full[i]),no_of_bins),histtype = 'step',color = 'blue',alpha = 0.7,weights=np.ones(len(outputs_full[i])) / len(outputs_full[i]))
                plt.hist(targets_full[i],label = 'Actual',bins = np.linspace(min(targets_full[i]),max(targets_full[i]),no_of_bins),histtype = 'step',color = 'orange',alpha = 0.7,weights=np.ones(len(targets_full[i])) / len(targets_full[i]))
                plt.ylabel('Number of Light Curves')
                #plt.xlabel(xlabels[i])
                plt.legend()
                plt.savefig(f'{save_path}/{band}/Params/{param_names[i]}_Hist.png')
                plt.close()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    if transfer_function:
        np.save(f'{save_path}/{band}/TF/training_curve.npy',training_curve)
        np.save(f'{save_path}/{band}/TF/val_curve.npy',val_curve)
    else:
        if len(param_names) == 1:
            np.save(f'{save_path}/{band}/Params/training_curve_{param_names[0]}.npy',training_curve)
            np.save(f'{save_path}/{band}/Params/val_curve_{param_name[0]}.npy',training_curve)
        else:
            np.save(f'{save_path}/{band}/Params/training_curve_params.npy',training_curve)
            np.save(f'{save_path}/{band}/Params/val_curve_params.npy',training_curve)
    if transfer_function:
        return model,test_loss,training_curve,val_curve
    else:
        return model,test_loss,training_curve,val_curve, outputs_full,std_full,targets_full,names_full,everything,weights,means,stds
