import os
import shutil
from main_new_OG import *
import sys
import torchvision
import torch
from torch import nn
import pickle

def move_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Move .npy files from source to target directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.npy'):
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            shutil.move(source_path, target_path)
            
def initialise_pca_model(channel_sizes=[27, 64, 64, 64, 64, 64, 64, 64, 64], source_subdir="64-pca"):
    
    # Define target directories
    std_target_dir = '/data/apassi1/std_coefs/'
    var_target_dir = '/data/apassi1/variance-coefs/'

    # Join target directories with the input subdir to create full source paths
    std_source_dir = os.path.join(std_target_dir, source_subdir)
    var_source_dir = os.path.join(var_target_dir, source_subdir)

    # Move all .npy files from the source_subdir to the main directory
    move_files(std_source_dir, std_target_dir)
    move_files(var_source_dir, var_target_dir)
    
    # Initialise PCA model with the given dimensions
    model = make_pca(channel_sizes)
    
    # Move all .npy files back to the source_subdir directory
    move_files(std_target_dir, std_source_dir)
    move_files(var_target_dir, var_source_dir)

    return model


def initialise_random_model(channel_sizes=[27, 64, 64, 64, 64, 64, 64, 64, 64], source_subdir="64-random"):

    target_dir = '/data/apassi1/std_coefs/'
    source_dir = os.path.join(target_dir, source_subdir)

    move_files(source_dir, target_dir)
    model = make_random(channel_sizes)
    move_files(target_dir, source_dir)

    return model

from main import * 

def initialise_backprop_model():
    path = '/data/apassi1/rainbow_models/Pr_Norm/batchsize_128_lrfreq_45_best.pth.tar'
    model = load_model() 
    checkpoint = torch.load(path)
    
    state_dict = checkpoint["state_dict"]
    state_dict = {key.replace("(0, 0)", "0"): value for key, value in state_dict.items()}
    checkpoint["state_dict"] = state_dict
    model.load_state_dict(checkpoint['state_dict'])
    return model

torch.manual_seed(0)
torch.cuda.manual_seed(0)
trained_model = torchvision.models.alexnet(pretrained=True) # LOADING TRAINED ALEXNET 
untrained_model = torchvision.models.alexnet(pretrained=False) # LOADING UNTRAINED ALEXNET 

class Model(nn.Module):

    def __init__(self,
                features_layer: str,
                gpool: bool=False,
                ):
        
        super(Model, self).__init__()        
        self.features_layer = features_layer
        self.gpool = gpool
        
        
    def forward(self, x):                
        # extract activations from 
        activation = {}
        def get_activation(name):
            def hook(trained_model, input, output):
                activation[name] = output.detach().cuda()
            return hook
        trained_model.features[self.features_layer].register_forward_hook(get_activation(f'features.{self.features_layer}'))
        trained_model.to("cuda")
        output = trained_model(x)        
        x = activation[f'features.{self.features_layer}']                       
        if self.gpool:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)        
        return x.flatten(start_dim=1,end_dim=-1)    
    
class Alexnet:
    def __init__(self, features_layer:str = 12, gpool:int = False):
        self.features_layer = features_layer
        self.gpool = gpool
    
    def Build(self):
        return Model(    
                features_layer = self.features_layer,
                gpool = self.gpool)
        
class UntrainedModel(nn.Module):
    def __init__(self,
                 features_layer: str,
                 gpool: bool = False):
        super(UntrainedModel, self).__init__()
        self.features_layer = features_layer
        self.gpool = gpool
        
    def forward(self, x):
        # extract activations from the specified layer
        activation = {}
        
        def get_activation(name):
            def hook(untrained_model, input, output):
                activation[name] = output.detach().cuda()
            return hook
        
        untrained_model.features[self.features_layer].register_forward_hook(get_activation(f'features.{self.features_layer}'))
        untrained_model.to("cuda")
        output = untrained_model(x)
        x = activation[f'features.{self.features_layer}']
        
        if self.gpool:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
        
        return x.flatten(start_dim=1, end_dim=-1)

class AlexnetUntrained:
    def __init__(self, features_layer: str = 12, gpool: int = False):
        self.features_layer = features_layer
        self.gpool = gpool
    
    def Build(self):
        return UntrainedModel(
            features_layer=self.features_layer,
            gpool=self.gpool
        )

def extract_alexnet_activations(images, training_status, layerid):

    if training_status:
        model = Alexnet(features_layer=layerid).Build()  # Trained model
    else:
        model = AlexnetUntrained(features_layer=layerid).Build()  # Untrained model

    model = model.cuda()

    batch_size = 50
    num_samples = len(images)
    x = []

    for i in range(0, num_samples, batch_size):
        batch_input = images[i:i + batch_size]
        batch_input = batch_input.cuda()
        with torch.no_grad():
            output = model(batch_input)
        
        output = output.cpu()
        x.append(output)

    x = torch.cat(x, dim=0)
    return x

def extract_alexnet_pooled_activations(images, training_status, layerid, gpool=True):
    # If training_status is True, we use the trained AlexNet model; otherwise, we use the untrained version.
    if training_status:
        model = Alexnet(features_layer=layerid, gpool=gpool).Build()  # Trained model
    else:
        model = AlexnetUntrained(features_layer=layerid, gpool=gpool).Build()  # Untrained model

    model = model.cuda()

    batch_size = 50
    num_samples = len(images)
    x = []

    for i in range(0, num_samples, batch_size):
        batch_input = images[i:i + batch_size]
        batch_input = batch_input.cuda()
        with torch.no_grad():
            output = model(batch_input)
        
        output = output.cpu()
        x.append(output)

    x = torch.cat(x, dim=0)
    return x