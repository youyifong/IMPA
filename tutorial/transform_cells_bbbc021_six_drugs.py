#!/usr/bin/env python
# coding: utf-8

# # Use IMPA to transform BBBC021 

# Import libraries 

# In[10]:


# Standard library imports
import os
from pathlib import Path

# Third-party library imports
from tutorial_utils import t2np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml 
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Local application/library imports
from IMPA.dataset.data_loader import CellDataLoader
from IMPA.solver import IMPAmodule


# Read the configuration of interest 

# In[11]:


path_to_config = "../config_hydra/config/bbbc021_six.yaml"


# In[12]:


# Reading the YAML file
with open(path_to_config, 'r') as file:
    config = yaml.safe_load(file)  # Use safe_load to avoid executing arbitrary code

# Access the loaded data
print(config)


# In[13]:


config["image_path"] = "../" + config["image_path"]
config["data_index_path"] = "../" + config["data_index_path"]
config["embedding_path"] = "../" + config["embedding_path"]


# Create an omega config dict

# In[14]:


args = OmegaConf.create(config)


# #### Initialize data loader

# In[15]:


dataloader = CellDataLoader(args)


# #### Initialize model 

# In[16]:


checkpoint_dir = "../checkpoints/bbbc021_six/"


# In[17]:


solver = IMPAmodule(args, checkpoint_dir, dataloader)
solver._load_checkpoint(200)


# **Check what molecules we have and their encodings**

# In[18]:


train_dataloader = dataloader.train_dataloader()
val_dataloader = dataloader.val_dataloader()


# In[19]:


dataloader.mol2id


# **Tranform controls to perturbed**

# In[20]:


# Initilize empty dictionaries 
controls = []
transformed = {}

with torch.no_grad():
    for i, (drug, drug_id) in enumerate(dataloader.mol2id.items()):
        print(f"Transforming images for {drug}")
        transformed[drug] = []
        for batch in tqdm(dataloader.train_dataloader()):
            X_ctr = batch["X"][0]
            # z original and z transported
            z = torch.randn(X_ctr.shape[0], 100, args.z_dimension).cuda().mean(1)
    
            # Perturbation ID 
            id_pert = dataloader.mol2id[drug] * torch.ones(X_ctr.shape[0]).long().cuda()
            y = solver.embedding_matrix(id_pert)
            y = torch.cat([y, z], dim=1)
            y = solver.nets.mapping_network(y)
    
            _, X_generated = solver.nets.generator(X_ctr, y)

            if i==0:
                controls.append(t2np(X_ctr.detach().cpu(), batch_dim=True))
            transformed[drug].append(t2np(X_generated.detach().cpu(), batch_dim=True))
            
controls = np.concatenate(controls, axis=0) 
transformed = {key: np.concatenate(val, axis=0) for key, val in transformed.items()}


# In[21]:


for i in range(len(controls)):
    print(f"Control {i}")
    plt.figure(figsize=(1, 1))
    plt.imshow(controls[i])
    plt.axis("off")
    plt.show()
    if i==3:
        break


# In[22]:


for pert in transformed:
    print(f"Perturbation {pert}")
    for i in range(len(transformed[pert])):
        plt.figure(figsize=(1, 1))
        plt.imshow(transformed[pert][i])
        plt.axis("off")
        plt.show()
        if i==3:
            break


# In[ ]:




