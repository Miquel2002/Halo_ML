# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: temps
#     language: python
#     name: temps
# ---

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
#rom torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import scipy as scp

import sys
sys.path.append('/nfs/pic.es/user/m/mgraell/Halo_ML')
from  funcions_v2 import *

# # LOAD DATA

cat = get_dataset('/data/astro/scratch/lcabayol/EUCLID/GHalo_FS/FS2_clusters_z04.parquet',
                  ['halo_id','color_kind','x_in_halo', 'y_in_halo','z_in_halo', 'radis', 'abs_mag_r01',
                   'g01r01_hod','ax_halo', 'ay_halo', 'az_halo','b_to_a_halo','c_to_a_halo'])
len(cat)/1e6

cat_gb = cat.groupby('halo_id')
y_list = cat_gb.first()[['b_to_a_halo', 'c_to_a_halo']].reset_index()
X_list = cat_gb[['x_in_halo', 'y_in_halo', 'z_in_halo', 'radis', 'abs_mag_r01', 'g01r01_hod']].apply(lambda x: x.values.tolist()).reset_index(name='properties')
X_list = X_list['properties'].tolist()
unique_y = y_list[['b_to_a_halo','c_to_a_halo']].to_numpy().tolist()

del cat, cat_gb

# ## FILTER DATA BASED ON NUMBER OF GALAXIES IN HALO

X_list_full = X_list.copy()
unique_y_full = unique_y.copy()

# +
filtered_pairs = [(x, y) for x, y in zip(X_list, unique_y) if 400 <= len(x) < 3000]

if filtered_pairs:
    X_list, unique_y = map(list, zip(*filtered_pairs))
else:
    X_list, unique_y = [], []

# +
unique_y_tensor = torch.tensor(np.array(unique_y))


counts, bins = torch.histogram(unique_y_tensor[:,1], bins=10)
plt.plot(bins.numpy()[1:], counts.numpy())
plt.title('Counts de c/a al catàleg') 
plt.show()
# -

# ## OVERSAMPLE UNDER-REPRESENTED OBJECTS

# +
moda = max(counts)

X_list_pes = X_list.copy()
unique_y_pes = list(unique_y_tensor)
for j in range(len(unique_y_tensor)):
    halo_shape = unique_y_tensor[j]
    halo_data = X_list[j]
    for i in range(len(counts)):
        if halo_shape[1] > bins[i] and halo_shape[1] < bins[i+1]:
            size= random_round(moda/(2*counts[i]) - 1/2)
            for k in range(size):
                insert_index =  np.random.randint(0,len(unique_y_pes))
                unique_y_pes.insert(insert_index, halo_shape)
                X_list_pes.insert(insert_index, halo_data)

unique_y_pes_tensor = unique_y_pes[0]
for i in range(1, len(unique_y)):
    unique_y_pes_tensor = torch.vstack((unique_y_pes_tensor, unique_y_pes[i]))


# -

counts, bins = torch.histogram(unique_y_pes_tensor[:,1], bins=10)
plt.plot(bins.numpy()[1:], counts.numpy())
plt.title('Counts de c/a després del "pesatge"')
plt.show()
unique_y_pes_tensor.shape, len(unique_y), len(unique_y_pes), len(X_list), len(X_list_pes)

# +
# Faig batches
_, X_test = make_test_train_batched(X_list)
_, y_test_list = make_test_train_batched(unique_y)

X_train, _ = make_test_train_batched(X_list_pes)
y_train_pes_list, _ = make_test_train_batched(unique_y_pes)

_, X_test_full = make_test_train_batched(X_list_full)
_, y_test_full = make_test_train_batched(unique_y_full)

# -

del X_list, unique_y, X_list_pes, unique_y_pes, X_list_full, unique_y_full, _

y_test = torch.tensor(np.array(y_test_list))
y_train = torch.tensor(np.array(y_train_pes_list))

X_test_analitic = X_test_full.copy()
for i in range(len(X_test_analitic)):
    for j in range(len(X_test_analitic[i])):
        halo = X_test_analitic[i][j]
        X_test_analitic[i][j] = [gal for gal in halo if gal[4] != 0]
    

# # DEFINE MODEL

# +
## in general, codes should not have copy() and del all... you have many
# -

X_test_tot = X_test_full.copy()
y_test_tot = y_test_full.copy()

del X_test_full, y_test_full, y_test_list, y_train_pes_list

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


class Model(nn.Module):
    def __init__(self, input_len:int, output_len:int, batch_size:int):
        super().__init__()
                        
        self.lin_block = nn.Sequential(
                                nn.Linear(input_len, 256),
                                nn.ELU(),
                                nn.Linear(256, 256),                                
                            )
        self.lin_block2 = nn.Sequential(
                                nn.Linear(256, 256),
                                nn.ELU(),
                                nn.Linear(256, output_len),
                            )
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len



    def forward(self, data):
        x = [self.lin_block(x_elem.to(device)) for x_elem in data]
        
        # xsum =[xx.sum(dim=0) for xx in x]
        # x = torch.stack(xsum)
        
        max_len = max([len(x_elem) for x_elem in data])

        padded_x = [torch.nn.functional.pad(x_elem, (0, 0, 0, max_len - len(x_elem))) for x_elem in x]

        stacked_data = torch.stack(padded_x)

        x = stacked_data.sum(dim=1)
        
        output = self.lin_block2(x)

        return output.squeeze()

torch.manual_seed(42)
input_len = 6
output_len = 2
model = Model(input_len=input_len, output_len=output_len, batch_size = 128).to(device)


# # TRAINING FUNCTION

# +
# Some changes:
    #- it is not very nice to have a function as an argument (loss_fn)
    #- you can have a single function for the train and test, it's simpler
    #- I ssume you are not changing the scheduler, so for the time beign I hardcoded the arguments

def training(model, xdata, ydata, xval, yval, nepochs=2500 ,lr=1e-2):
    opt = torch.optim.Adagrad(model.parameters(), 
                              lr=lr)
    loss_fn = nn.L1Loss()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 
    #                                                       mode='min', 
    #                                                       factor=0.2, 
    #                                                       patience=300, 
    #                                                       threshold=0.0001, 
    #                                                       threshold_mode='abs')
    
    for epoch in range(nepochs):
        train_loss, val_loss = 0,0
        for i in range(len(xdata)):

            opt.zero_grad()
            output = model(xdata[i]).to(device)
            
            loss = loss_fn(output, ydata[i].to(device))
            loss.backward()
            opt.step()
    
            train_loss += loss.item()

        print(output)
    
        train_loss /= len(xdata)

        for j in range(len(xval)):
            output = model(xval[j]).to(device)
            loss = loss_fn(output, yval[j].to(device))
            val_loss += loss.item()
        val_loss /= len(xval)

        print(f'Train loss: {train_loss:.5f}', f'Val loss: {val_loss:.5f}')
    
            
        
# -

# # USING OPTUNA

import optuna
from optuna.trial import TrialState


def optimizer_function(trial):
    model = Model(input_len=input_len, 
                  output_len=output_len, 
                  batch_size = trial.suggest_int("batch_size",100,200,step=100)).to(device)

    training(model, X=X_train, 
             y=y_train,
             xval=X_test, 
             yval=y_test, 
             nepochs=trial.suggest_int("nepochs",100,600, step=100),
             lr=trial.suggest_float("lr",1e-5,1e-3,step=1e-4))

    ## add here a function to evaluate accuracy or loss or whatever metric you consider the most important
    ## e.g. err = F()

    return err
    


study = optuna.create_study(direction="minimize")

study.optimize(optimizer_function, n_trials=100)


# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Anàlisi
# -

def get_loss_acc(model, X, y, loss_fn, exigencia):
    model.eval()
    test_loss = 0
    if output_len > 1:
        acc = np.zeros(output_len)
    else:
        acc = 0
    with torch.inference_mode():
        for i in range(len(X)):
            output = model(X[i]).to(device)
            labels = y[i].to(device)
            test_loss += loss_fn(output, labels)
            encerts = 0
            if output_len > 1:
                for j in range(len(output)):  
                    rel_err = torch.abs(output[j]-labels[j])*100/labels[j] 
                    for k in range(output_len):    
                        if rel_err[k] < exigencia:
                            acc[k] += 1/len(output)
            else:
                for j in range(len(output)):  
                    rel_err = torch.abs(output[j]-labels[j])*100/labels[j] 
                    if rel_err < exigencia:
                        acc += 1/len(output)                

        test_loss /= len(X)
        acc *= 100/len(X) 
        print(f'Test loss: {test_loss: .5f}, Accuracy: {np.round(acc, 2)} %')
        return test_loss, acc


# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import scipy as scp
from funcions_v2 import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class Model(nn.Module):
    def __init__(self, input_len:int, output_len:int, batch_size:int):
        super().__init__()
                        
        self.lin_block = nn.Sequential(
                                nn.Linear(input_len, 256),
                                nn.ELU(),
                                nn.Linear(256, 256),                                
                            )
        self.lin_block2 = nn.Sequential(
                                nn.Linear(256, 256),
                                nn.ELU(),
                                nn.Linear(256, output_len),
                            )
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len



    def forward(self, data):
        x = [self.lin_block(x_elem.to(device)) for x_elem in data]
        # xsum =[xx.sum(0) for xx in x]
        # x = torch.stack(xsum)
        
        max_len = max([len(x_elem) for x_elem in data])

        padded_x = [torch.nn.functional.pad(x_elem, (0, 0, 0, max_len - len(x_elem))) for x_elem in x]

        stacked_data = torch.stack(padded_x)

        x = stacked_data.sum(dim=1)
        
        output = self.lin_block2(x)

        return output.squeeze()
    
torch.manual_seed(42)
input_len = 6
output_len = 2
model = Model(input_len=input_len, output_len=output_len, batch_size = 128).to(device)
model.load_state_dict(torch.load('/nfs/pic.es/user/m/mgraell/Halo_ML/bctoa_model_blau_0.pt'))

print(len(X_test_tot), len(X_test_analitic), y_test_tot.shape)
# -

for i in range(len(X_test_tot)):
    for j in range(len(X_test_tot[i])):
        X_test_tot[i][j] = torch.tensor(np.array(X_test_tot[i][j]))
        X_test_analitic[i][j] = torch.tensor(np.array(X_test_analitic[i][j]))

# +
model.eval()

labels_tot_list = []
preds_tot_list = []
preds_analitic_tot_list = []


nan_index0 = []
nan_index1 = []
with torch.inference_mode():
    preds_tot = np.array([model(batch).cpu().numpy() for batch in X_test_tot])
    labels_tot = y_test_tot.cpu().numpy()
    print(preds_tot.shape)
    print(labels_tot.shape)
    preds_tot = preds_tot.reshape(preds_tot.shape[0]*preds_tot.shape[1], preds_tot.shape[2])
    labels_tot = labels_tot.reshape(labels_tot.shape[0]*labels_tot.shape[1], labels_tot.shape[2])
    for i in range(len(X_test_analitic)):
        preds_analitic = []
        for j in range(len(X_test_analitic[i])):
            computeshape3d = compute_shape_3D(X_test_analitic[i][j][:,0].numpy(),X_test_analitic[i][j][:,1].numpy(),
                                              X_test_analitic[i][j][:,2].numpy(), 'standard')
            preds_analitic.append(np.array([computeshape3d[1][1]/computeshape3d[1][0], computeshape3d[1][2]/computeshape3d[1][0]]))
        preds_analitic_tot_list.append(np.array(preds_analitic))
    preds_analitic_tot = np.array(preds_analitic_tot_list)
    print(preds_analitic_tot.shape)
    preds_analitic_tot = preds_analitic_tot.reshape(preds_analitic_tot.shape[0]*preds_analitic_tot.shape[1], preds_analitic_tot.shape[2])

    print(preds_analitic_tot.shape)
    
    preds_analitic_btoa = preds_analitic_tot[:,0]
    preds_analitic_ctoa = preds_analitic_tot[:,1]

    for i in range(len(preds_analitic_btoa)):
        if np.isnan(preds_analitic_btoa[i]):
            nan_index0.append(i)
    inds_to_delete0 = sorted(nan_index0, reverse=True)
    for i in range(len(preds_analitic_ctoa)):
        if np.isnan(preds_analitic_ctoa[i]):
            nan_index1.append(i)
    inds_to_delete1 = sorted(nan_index1, reverse=True)
    
    labels_analitic_btoa = labels_tot[:,0]
    labels_analitic_ctoa = labels_tot[:,1]
    
    for i in inds_to_delete0:
        labels_analitic_btoa = np.delete(labels_analitic_btoa, i)
        preds_analitic_btoa = np.delete(preds_analitic_btoa, i)
    for i in inds_to_delete1:
        labels_analitic_ctoa = np.delete(labels_analitic_ctoa, i)
        preds_analitic_ctoa = np.delete(preds_analitic_ctoa, i)

    
    print(labels_tot.shape, preds_tot.shape, labels_analitic_btoa.shape, preds_analitic_btoa.shape)
    

    corr_btoa = np.corrcoef(labels_tot[:,0], preds_tot[:,0])[0,1]
    corr_ctoa = np.corrcoef(labels_tot[:,1], preds_tot[:,1])[0,1]
    corr_analitic_btoa = np.corrcoef(labels_analitic_btoa, preds_analitic_btoa)[0,1]
    corr_analitic_ctoa = np.corrcoef(labels_analitic_ctoa, preds_analitic_ctoa)[0,1]
    
    

    
    
    plt.plot(labels_tot[:,0], preds_tot[:,0], '.', color = 'r', markersize = .5, label='Model predictions')
    plt.plot(labels_analitic_btoa, preds_analitic_btoa, '.', color = 'g', markersize = .5, label='Analytic calculation')
    plt.title('b/a')
    plt.legend()



    print('b/a r2:', np.round(corr_btoa, 2), '|', 'c/a r2:', np.round(corr_ctoa, 2))
    print('Analytic b/a r2:', np.round(corr_analitic_btoa, 2), '|', 'Analytic c/a r2:', np.round(corr_analitic_ctoa, 2))
    plt.plot(np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]), '--', color = 'cornflowerblue')

    plt.xlim(0.3,1)
    plt.ylim(0.3,1)
    plt.show()

    plt.plot(labels_tot[:,1], preds_tot[:,1], '.', color = 'r', markersize = .5, label='Model predictions')
    plt.plot(labels_analitic_ctoa, preds_analitic_ctoa, '.', color = 'g', markersize = .5, label='Analytic calculation')
    plt.title('c/a')
    plt.legend()


print('b/a r2:', np.round(corr_btoa, 2), '|', 'c/a r2:', np.round(corr_ctoa, 2))
print('Analytic b/a r2:', np.round(corr_analitic_btoa, 2), '|', 'Analytic c/a r2:', np.round(corr_analitic_ctoa, 2))
plt.plot(np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]), '--', color = 'cornflowerblue')

plt.xlim(0.3,1)
plt.ylim(0.3,1)
plt.show()




bins_number = 20


binstat = scp.stats.binned_statistic(labels_tot[:,0], preds_tot[:,0], statistic='mean', bins=bins_number, range=None)
preds_bins = binstat[0]
labels_bins = binstat[1]
labels_bins = np.delete(labels_bins, 0)

binstat_analitic = scp.stats.binned_statistic(labels_analitic_btoa, preds_analitic_btoa, statistic='mean', bins=bins_number, range=None)
preds_analitic_bins = binstat_analitic[0]

plt.plot(labels_bins, preds_bins, '-', color = 'r', label='Model predictions')
plt.plot(labels_bins, preds_analitic_bins, '-', color = 'g', label='Analytic calculation')
plt.title('b/a, binned')
plt.legend()

plt.plot([0.3,1],[0.3,1], '--', color = 'cornflowerblue')
plt.show()

binstat = scp.stats.binned_statistic(labels_tot[:,1], preds_tot[:,1], statistic='mean', bins=bins_number, range=None)
preds_bins = binstat[0]
labels_bins = binstat[1]
labels_bins = np.delete(labels_bins, 0)

binstat_analitic = scp.stats.binned_statistic(labels_analitic_ctoa, preds_analitic_ctoa, statistic='mean', bins=bins_number, range=None)
preds_analitic_bins = binstat_analitic[0]

plt.plot(labels_bins, preds_bins, '-', color = 'r', label='Model predictions')
plt.plot(labels_bins, preds_analitic_bins, '-', color = 'g', label='Analytic calculation')
plt.title('c/a, binned')
plt.legend()

plt.plot([0.3,1],[0.3,1], '--', color = 'cornflowerblue', linewidth = '1')
plt.show()



