import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
#import networkx as nx
#from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

def random_round(numero):
    num_int = int(numero)
    decimal = numero - num_int
    num_al = np.random.uniform()
    if num_al <= decimal:
        num_aprox = num_int + 1
    else:
        num_aprox = num_int
    return(num_aprox)



def get_dataset(file_path, dades):
    '''
    Agafa el cataleg en parquet retorna un dataframe amb halos
    d'entre lim_inf (inclos) i lim_sup (no inclos) galaxies amb les columnes que ens interessen
    '''
    cat = pd.read_parquet(file_path) #obrir catàleg
    array_radis = np.sqrt((cat['x_gal'].to_numpy() - cat['x_halo'].to_numpy())** 2  #distància de les galàxies respecte del centre
                      + (cat['y_gal'].to_numpy() - cat['y_halo'].to_numpy()) ** 2
                      + (cat['z_gal'].to_numpy() - cat['z_halo'].to_numpy()) ** 2)
    cat['radis'] = pd.DataFrame(array_radis)
    array_x = cat['x_gal'].to_numpy() - cat['x_halo'].to_numpy() #posicions respecte del centre dels halos, per tenir dades més diferenciades per a millor anàlisi del model
    array_y = cat['y_gal'].to_numpy() - cat['y_halo'].to_numpy()
    array_z = cat['z_gal'].to_numpy() - cat['z_halo'].to_numpy()
    cat['x_in_halo'] = pd.DataFrame(array_x)
    cat['y_in_halo'] = pd.DataFrame(array_y)
    cat['z_in_halo'] = pd.DataFrame(array_z)

    cat = cat[dades]

    return cat


def make_batches(data, batch_size, drop_last):
    '''
    Converteix l'input (llista de tensors) en una llista de batches (llistes) de tensors
    '''
    output = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i : i+batch_size]
        tensor_list = [torch.tensor(sample).to(torch.float32) for sample in batch]
        output.append(tensor_list)
    
    if drop_last:
        output = output[:-1]
    return output


def make_test_train_batched(data, train_fraction = 0.8, batch_size = 128, drop_last = True):
    '''
    Converteix l'input (llista de tensors) en dues (train i test) llistes de batches (llistes) de tensors 
    '''
    train_size = int(train_fraction * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    train_output = make_batches(train_data, batch_size, drop_last=drop_last)
    test_output = make_batches(test_data, batch_size, drop_last=drop_last)

    return train_output, test_output

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

def train_step(model, X, y, loss_fn, opt, scheduler, use_scheduler, epoch, print_every):
    train_loss = 0
    model.train()
    
    for i in range(len(X)):
        opt.zero_grad()
        output = model(X[i]).to(device)
        

        loss = loss_fn(output, y[i].to(device))
        loss.backward()
        opt.step()

        train_loss += loss.item()

    train_loss /= len(X)
    if epoch % print_every == 0:
        print(f'Train loss: {train_loss:.5f}')
    if use_scheduler:
        scheduler.step(loss)


def test_step(model, X, y, loss_fn, epoch, print_every, epochs, exigencia):
    if epoch % print_every == 0:
        test_loss, acc = get_loss_acc(model, X, y, loss_fn, exigencia)
        print(np.round(epoch/epochs *100, 1), '%')


def projected_coodinates(x,y,z,xc,yc,zc):

    ra_center = np.arctan(xc/yc)
    dec_center = np.arcsin(zc/np.sqrt(xc**2 + yc**2 + zc**2))
     
    e1x =   np.cos(ra_center)
    e1y =   -np.sin(ra_center)
     
    e2x = -np.sin(dec_center) * np.sin(ra_center)
    e2y = -np.sin(dec_center) * np.cos(ra_center)
    e2z =  np.cos(dec_center)
     
     # Projected coordinates
    xp = e1x*x + e1y*y
    yp = e2x*x + e2y*y + e2z*z
    
    return xp,yp


def compute_shape_3D(x,y,z,tensor_def='reduced'):
     
    '''

    Compute 3D and projected 2D axis
    according to the moment of inertia
    (INPUT)
    x,y,z 3D coordinates arrays of len(N)
       wher N is the total number of particles

    tensor_def = 'reduced': compute reduced tensor
            = 'standard': compute standard tensor
    (OUTPUT)
    v3,w3d Eigenvectors and sqrt(eingenvalues) in 3D
    '''

     
    if tensor_def == 'reduced':
        w = 1./(x**2+y**2+z**2)

    if tensor_def == 'standard':
        w = np.ones(len(x))

     # COMPUTE 3D Tensor

    T3D = np.zeros((3,3))

    T3D[0,0] = np.sum(w*x**2)/len(x)
    T3D[0,1] = np.sum(w*x*y)/len(x)
    T3D[0,2] = np.sum(w*x*z)/len(x)

    T3D[1,0] = np.sum(w*y*x)/len(x)
    T3D[1,1] = np.sum(w*y**2)/len(x)
    T3D[1,2] = np.sum(w*y*z)/len(x)

    T3D[2,0] = np.sum(w*z*x)/len(x)
    T3D[2,1] = np.sum(w*z*y)/len(x)
    T3D[2,2] = np.sum(w*z**2)/len(x)

    w3d,v3d =np.linalg.eig(T3D)

    j = np.flip(np.argsort(w3d))
    w3d = w3d[j] # Ordered eingenvalues
    v3d = v3d[:,j] # Ordered eingenvectors
     
    return v3d,np.sqrt(w3d)
     
     # -----------------------------------------------
     # COMPUTE projected quantities
         
    T2D = np.zeros((2,2))
     
    T2D[0,0] = np.sum(wp*xp**2)
    T2D[0,1] = np.sum(wp*xp*yp)
    T2D[1,0] = np.sum(wp*xp*yp)
    T2D[1,1] = np.sum(wp*yp**2)
     
    w2d,v2d =np.linalg.eig(T2D)
     
    j = np.flip(np.argsort(w2d))
    w2d = w2d[j] # Ordered eingenvalues
    v2d = v2d[:,j] # Ordered eingenvectors
     
    return v3d,np.sqrt(w3d),v2d,np.sqrt(w2d)


def get_dataset_analitic(file_path):
    cat = pd.read_parquet(file_path) #obrir catàleg
    array_radis = np.sqrt((cat['x_gal'].to_numpy() - cat['x_halo'].to_numpy())** 2  #distància de les galàxies respecte del centre
                      + (cat['y_gal'].to_numpy() - cat['y_halo'].to_numpy()) ** 2
                      + (cat['z_gal'].to_numpy() - cat['z_halo'].to_numpy()) ** 2)
    cat['radis'] = pd.DataFrame(array_radis)
    array_x = cat['x_gal'].to_numpy() - cat['x_halo'].to_numpy() #posicions respecte del centre dels halos, per tenir dades més diferenciades per a millor anàlisi del model
    array_y = cat['y_gal'].to_numpy() - cat['y_halo'].to_numpy()
    array_z = cat['z_gal'].to_numpy() - cat['z_halo'].to_numpy()
    cat['x_in_halo'] = pd.DataFrame(array_x)
    cat['y_in_halo'] = pd.DataFrame(array_y)
    cat['z_in_halo'] = pd.DataFrame(array_z)
    
    cat_analitic = cat[cat['radis']!=0]
    cat_analitic = cat_analitic[['halo_id','x_in_halo', 'y_in_halo','z_in_halo']]#, 'radis','ax_halo', 'ay_halo', 'az_halo','b_to_a_halo','c_to_a_halo']]

    counts = cat_analitic['halo_id'].value_counts()
    
    halos_n_gal_series = list(counts.index)

    cat2 = cat_analitic[cat_analitic['halo_id'].isin(halos_n_gal_series)]
    return cat2
