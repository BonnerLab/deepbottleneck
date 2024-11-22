import os
import numpy as np
import torch.nn as nn
from main_new import *
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import numpy as np
import random
from sklearn.decomposition import IncrementalPCA
from main import *
import h5py
import xarray as xr
from torch_cv import *
from regression import *
from sklearn.linear_model import Ridge
import shutil

def generate_model_to_compute_mean_and_var(num_channels):

    model = load_model_modified(num_channels)
    path = '/data/apassi1/variance-coefs'
    n = len([f for f in os.listdir(path) if f.startswith('bias') and f.endswith('.npy')])
    if n > 0: 
        new_num_channels = num_channels[:n]
        m1 = make_pca(new_num_channels)
        new_model = nn.Sequential(
            model[0], 
            *[m1[i] for i in range(1, len(new_num_channels) + 1)], 
            model[len(new_num_channels) + 1][:2]
            )
    else: 
        new_model = nn.Sequential(
            model[0], 
            model[1][:2]
            )
    
    return new_model

def generate_model_to_compute_pm_and_bias(num_channels):

    mean_files_dir = '/data/apassi1/std_coefs'
    mean_files = [f for f in os.listdir(mean_files_dir) if f.startswith('mean') and f.endswith('.npy')]
    n = len(mean_files)
    model = load_model_new(num_channels[:n])
    
    if n > 1: 
        m1 = make_pca(num_channels[:n-1])
        new_model = nn.Sequential(
            model[0],  # First layer from model
            *[m1[i] for i in range(1, len(num_channels[:n-1]) + 1)],  # Layers from m1
            model[len(num_channels[:n-1]) + 1][:3]  # Final layers from model
            )
    else: 
        new_model = nn.Sequential(
            model[0],  # First layer from model
            model[1][:3]  # Final layers from model
            )
    
    return new_model

def train_model(channel_sizes, num_training_images, seed=42):
    # Step 1: Select a subsection of training images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    # Step 2: Loop over the number of layers in the model
    for layerid in range(1, len(channel_sizes) + 1):

        # Step 6a: Initialize a model for mean and variance computation
        model = generate_model_to_compute_mean_and_var(channel_sizes)
        model.cuda()

        # Step 6b: Compute mean
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        # Step 6b: Compute variance
        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, dim=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)

        m = m.cpu().squeeze()
        std = std.cpu()

        # Step 6c: Save mean and variance
        mean_path = f'/data/apassi1/std_coefs/mean{layerid}.npy'
        var_path = f'/data/apassi1/std_coefs/var{layerid}.npy'
        np.save(mean_path, m.numpy())
        np.save(var_path, std.numpy())

        # Step 7a: Initialize a model for pm and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes)
        model.cuda()

        # Step 7b: Compute pm and bias
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        # Perform PCA
        pca = IncrementalPCA()
        pca.fit(coefs)

        # Get the projection matrix (pm) and bias
        projection_matrix = pca.components_
        projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)

        bias = projection_matrix @ channel_mean
        bias = -1 * bias

        # Step 7c: Save pm and bias
        pm_path = f'/data/apassi1/variance-coefs/pm_cent{layerid}.npy'
        bias_path = f'/data/apassi1/variance-coefs/bias{layerid}.npy'
        np.save(pm_path, projection_matrix.numpy())
        np.save(bias_path, bias.numpy())

        print(f"Layer {layerid} completed.")
        
def compute_encoding_score(model, data_array_unshared, data_array_shared, y_unshared, y_shared, batch_size=10):
    # Ensure model is on GPU
    model = model.cuda()
    
    num_samples_train = data_array_unshared.size(0)
    num_samples_test = data_array_shared.size(0)
    
    # Extracting training set
    xtrain = []
    xtest = []
    
    for i in range(0, num_samples_train, batch_size):
        batch_input = data_array_unshared[i:i + batch_size]
        batch_input = batch_input.cuda()

        with torch.no_grad():
            output = model(batch_input)
            output = output.full_view()  # Assuming full_view is a custom method
            output = output.reshape(output.shape[0], -1)
            output = output.cpu()
            xtrain.append(output)

    xtrain = torch.cat(xtrain, dim=0)
    
    # Extracting test set
    for i in range(0, num_samples_test, batch_size):
        batch_input = data_array_shared[i:i + batch_size]
        batch_input = batch_input.cuda()

        with torch.no_grad():
            output = model(batch_input)
            output = output.full_view()
            output = output.reshape(output.shape[0], -1)
            output = output.cpu()
            xtest.append(output)

    xtest = torch.cat(xtest, dim=0)

    # Perform regression
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
    
    regression.fit(xtrain, y_unshared)
    best_alpha = float(regression.alpha_)
    
    # Perform regression
    y_true, y_predicted = regression_shared_unshared(
        x_train=xtrain,
        x_test=xtest,
        y_train=y_unshared,
        y_test=y_shared,
        model=Ridge(alpha=best_alpha),
    )
    
    y_true = y_true.T
    y_predicted = y_predicted.T

    r2 = torch.stack([pearson_r(y_true_, y_predicted_)
                      for y_true_, y_predicted_ in zip(y_true, y_predicted)])
    
    e = r2.mean()
    
    return e

def train_model_new(channel_sizes, num_training_images):
    # Step 1: Select a subsection of training images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
        
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    
    # Calculate n as the floor of num_training_images / 1000
    n = num_training_images // 1000

    # Dictionary to store selected indices from each class/folder
    folder_to_indices = {}

    # Loop through the classes (subfolders)
    for class_idx, folder_name in enumerate(dataset.classes):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        
        # Select n random images from the current class/folder
        selected_indices = random.sample(class_indices, min(n, len(class_indices)))
        folder_to_indices[class_idx] = selected_indices

    # Combine the selected indices from all folders into a single list
    subset_indices = [idx for indices in folder_to_indices.values() for idx in indices]
    dataset = Subset(dataset, subset_indices)
    
    bs = 30
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    # Step 2: Loop over the number of layers in the model
    for layerid in range(1, len(channel_sizes) + 1):

        # Step 6a: Initialize a model for mean and variance computation
        model = generate_model_to_compute_mean_and_var(channel_sizes)
        model.cuda()

        # Step 6b: Compute mean
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / len(subset_indices)) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        # Step 6b: Compute variance
        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / len(subset_indices)) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, dim=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)

        m = m.cpu().squeeze()
        std = std.cpu()

        # Step 6c: Save mean and variance
        mean_path = f'/data/apassi1/std_coefs/mean{layerid}.npy'
        var_path = f'/data/apassi1/std_coefs/var{layerid}.npy'
        np.save(mean_path, m.numpy())
        np.save(var_path, std.numpy())

        # Step 7a: Initialize a model for pm and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes)
        model.cuda()

        # Step 7b: Compute pm and bias
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / len(subset_indices)) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        # Perform PCA
        pca = IncrementalPCA()
        pca.fit(coefs)

        # Get the projection matrix (pm) and bias
        projection_matrix = pca.components_
        projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)

        bias = projection_matrix @ channel_mean
        bias = -1 * bias

        # Step 7c: Save pm and bias
        pm_path = f'/data/apassi1/variance-coefs/pm_cent{layerid}.npy'
        bias_path = f'/data/apassi1/variance-coefs/bias{layerid}.npy'
        np.save(pm_path, projection_matrix.numpy())
        np.save(bias_path, bias.numpy())

        print(f"Layer {layerid} completed.")