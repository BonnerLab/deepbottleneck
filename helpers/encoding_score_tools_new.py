import xarray as xr
import numpy as np
from main import *
import h5py
from main_new import *
from torch_cv import *
from regression import *
from sklearn.linear_model import Ridge
from helpers.initialise_models import *
import torch
import os
from dotenv import load_dotenv
load_dotenv()
from bonner.models.hooks import *
from bonner.computation.rsa import *

def select_imageids(subjid):
    
    file_path = f"/data/apassi1/for_atlas/midventral visual stream/preprocessed/subject={subjid}.nc"
    dataset1 = xr.open_dataset(file_path)
    imageids1 = np.unique(dataset1["stimulus"])

    if subjid == 0: 
        file_path = f"/data/apassi1/for_atlas/midventral visual stream/preprocessed/subject=1.nc"
    else: 
        file_path = f"/data/apassi1/for_atlas/midventral visual stream/preprocessed/subject=0.nc"
    dataset2 = xr.open_dataset(file_path)
    imageids2 = np.unique(dataset2["stimulus"])

    imageids_unshared = sorted(list(set(imageids1) - set(imageids2)))
    imageids_shared = sorted(np.intersect1d(imageids1, imageids2))    
    return imageids_unshared, imageids_shared

def get_neural_activations(subjid, roi):

        file_path = f"/data/apassi1/for_atlas/{roi}/preprocessed/subject={subjid}.nc"
        ventral = xr.open_dataarray(file_path)
        
        x = list(map(str, ventral['x'].values))
        y = list(map(str, ventral['y'].values))
        z = list(map(str, ventral['z'].values))
        
        xyz_ventral = [f"{xi}--{yi}--{zi}" for xi, yi, zi in zip(x, y, z)]
        ventral = ventral.assign_coords(neuroid=xyz_ventral)
        
        general = xr.open_dataarray(f"/data/apassi1/for_atlas/roi=general/preprocessed/z_score=session.average_across_reps=True/subject={subjid}.nc")
        
        x = list(map(str, general['x'].values))
        y = list(map(str, general['y'].values))
        z = list(map(str, general['z'].values))
        
        xyz_general = [f"{xi}--{yi}--{zi}" for xi, yi, zi in zip(x, y, z)]
        general = general.assign_coords(neuroid=xyz_general)
        intersection = list(set(xyz_general) & set(xyz_ventral))
        dataset = ventral.sel(neuroid=intersection)
        y = dataset.sortby("stimulus").data
        
        imageids_unshared, imageids_shared = select_imageids(subjid)
        mask_unshared = np.isin(dataset.stimulus.astype(str), imageids_unshared)
        mask_shared = np.isin(dataset.stimulus.astype(str), imageids_shared) 
           
        return y[mask_unshared], y[mask_shared]

def load_images(subjid):

    imageids_unshared, imageids_shared = select_imageids(subjid)
    imageids_unshared = [int(image_id.replace('image', '')) for image_id in imageids_unshared]
    imageids_shared = [int(image_id.replace('image', '')) for image_id in imageids_shared]
    hdf5_file_path = '/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataset_name = 'imgBrick'
        data_array_unshared = hdf5_file[dataset_name][imageids_unshared]
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        dataset_name = 'imgBrick'
        data_array_shared = hdf5_file[dataset_name][imageids_shared]
    
    data_array_unshared = preprocess(data_array_unshared)
    data_array_shared = preprocess(data_array_shared)
    return data_array_unshared, data_array_shared

def extract_activations(model, layerid, data_array_unshared, data_array_shared):
    
    mmodel = model[:(layerid+1)]
    mmodel = mmodel.cuda()

    batch_size = 30
    num_samples_train = data_array_unshared.size(0)
    num_samples_test = data_array_shared.size(0)

    xtrain = []
    xtest = []
    
    for i in range(0, num_samples_train, batch_size):
        batch_input = data_array_unshared[i:i + batch_size]
        batch_input = batch_input.cuda()

        with torch.no_grad():
            output = mmodel(batch_input)
        output = output.full_view()  # Assuming full_view is a method on the model output
        output = output.reshape(output.shape[0], -1)
        output = output.cpu()
        xtrain.append(output)

    xtrain = torch.cat(xtrain, dim=0)

    for i in range(0, num_samples_test, batch_size):
        batch_input = data_array_shared[i:i + batch_size]
        batch_input = batch_input.cuda()

        with torch.no_grad():
            output = mmodel(batch_input)
        output = output.full_view()  # Assuming full_view is a method on the model output
        output = output.reshape(output.shape[0], -1)
        output = output.cpu()
        xtest.append(output)

    xtest = torch.cat(xtest, dim=0)
    
    mmodel = mmodel.cpu()
    return xtrain, xtest

def compute_encoding_score(xtrain, xtest, ytrain, ytest, epsilon = 0.1):
    
    n_samples = xtrain.shape[0] + xtest.shape[0]
    n_features = xtrain.shape[1]
    n_components = compute_johnson_lindenstrauss_limit(n_samples=n_samples, epsilon=epsilon)
    
    sparse_random_projection = SparseRandomProjection(
        n_components=n_components,
        density=None,
        seed=0,
        allow_expansion=False
    )
    
    xtrain = sparse_random_projection(xtrain)
    xtest = sparse_random_projection(xtest)

    xtrain = xtrain.cpu()
    xtest = xtest.cpu()
        
    ALPHA_RANGE = [10**i for i in range(10)]
    regression = TorchRidgeGCV(
        alphas=ALPHA_RANGE,
        fit_intercept=True,
        scale_X=False,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=False,
        device='cpu'
    )
    
    regression.fit(xtrain, ytrain)
    best_alpha = float(regression.alpha_)
    
    y_true, y_predicted = regression_shared_unshared(
        x_train=xtrain,
        x_test=xtest,
        y_train=ytrain,
        y_test=ytest,
        model=Ridge(alpha=best_alpha),
    )
    
    y_true = y_true.T
    y_predicted = y_predicted.T

    r2 = torch.stack([pearson_r(y_true_, y_predicted_)
                      for y_true_, y_predicted_ in zip(y_true, y_predicted)])
        
    e = r2.mean()
    return e

def compute_rsa(x, y):
    
    x = torch.tensor(x)
    y = torch.tensor(y)
    
    rsm_x = compute_rsm(x)
    rsm_y = compute_rsm(y)
    
    rsm_x = rsm_x.cuda()
    rsm_y = rsm_y.cuda()
    
    correlation_type = "Spearman"
    rsa_result = compute_rsa_correlation(rsm_x, rsm_y, correlation=correlation_type)
    
    return rsa_result

def compute_srp(xtrain, xtest, epsilon=0.1):

    n_samples = xtrain.shape[0] + xtest.shape[0]
    n_features = xtest.shape[1]
    
    n_components = compute_johnson_lindenstrauss_limit(n_samples=n_samples, epsilon=epsilon)   
     
    sparse_random_projection = SparseRandomProjection(
        n_components=n_components,
        density=None,
        seed=0,
        allow_expansion=False
    )
    
    xtrain = sparse_random_projection(xtrain)
    xtest = sparse_random_projection(xtest)
    
    return xtrain, xtest