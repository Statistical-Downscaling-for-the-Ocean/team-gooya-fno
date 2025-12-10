#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main by 
@author: rlc001
"""

import pandas as pd
import numpy as np
import glob
import os
import xarray as xr
import json
from pathlib import Path

def load_ctd_data(fpath,start_year, end_year):
    ds=xr.open_dataset(fpath)
    #start_date = pd.Timestamp(f"{start_year}-01-01")
    #end_date = pd.Timestamp(f"{end_year+1}-01-01")
    #ds = ds.where((ds.time >= start_date) & (ds.time < end_date), drop = True)
    return ds
    
def load_model_data(fpath, start_year, end_year):
    ds=xr.open_dataset(fpath)
    #start_date = pd.Timestamp(f"{start_year}-01-01")
    #end_date = pd.Timestamp(f"{end_year+1}-01-01")
    #ds = ds.where((ds.time >= start_date) & (ds.time < end_date), drop = True) #
    return ds, ds['bathy'].values

def normalize_dataset(ds, var_methods=None):
    """
    Normalize selected variables in an xarray.Dataset for ML.
    Returns:
      - normalized dataset
      - dictionary of scaling parameters for rescaling later
    """

    ds_norm = ds.copy(deep=True)
    scale_params = {}

    # Default normalization methods (can override with var_methods)
    default_methods = {
        "Temperature": "zscore",
        "t_pot": "zscore",
        "Salinity": "minmax",
        "Oxygen": "zscore",
        "Bathymetry": "minmax",
        "Depth": "minmax",
        "Latitude": None,
        "Longitude": None,
        "DOY" : None
    }

    if var_methods is None:
        var_methods = default_methods

    for var in ds.data_vars:
        method = var_methods.get(var, None)
        data = ds[var]

        if method == "zscore":
            mean_val = float(data.mean(skipna=True))
            std_val = float(data.std(skipna=True))
            ds_norm[var] = (data - mean_val) / std_val

            scale_params[var] = {
                "method": "zscore",
                "mean": mean_val,
                "std": std_val
            }

        elif method == "minmax":
            min_val = float(data.min(skipna=True))
            max_val = float(data.max(skipna=True))
            ds_norm[var] = (data - min_val) / (max_val - min_val)

            scale_params[var] = {
                "method": "minmax",
                "min": min_val,
                "max": max_val
            }

        else:
            # Variable not normalized (e.g., coordinates)
            scale_params[var] = {"method": None}
            continue

        print(f"Normalized {var} using {method}")

    return ds_norm, scale_params

def apply_normalization(ds, scale_params):
    """Apply precomputed normalization parameters to a dataset."""
    ds_norm = ds.copy(deep=True)
    for var, params in scale_params.items():
        if params["method"] == "zscore":
            mean_val = params["mean"]
            std_val = params["std"]
            ds_norm[var] = (ds[var] - mean_val) / std_val

        elif params["method"] == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            ds_norm[var] = (ds[var] - min_val) / (max_val - min_val)
        # else: leave unchanged
    return ds_norm

def reshape_to_tcsd(ds_input: xr.DataArray, ds_target: xr.DataArray):    ##NEW
    ds_input = xr.concat([ds_input[var] for var in list(ds_input.data_vars)], dim = 'channels')
    ds_target = xr.concat([ds_target[var] for var in list(ds_target.data_vars)], dim = 'channels')
    mask = (~np.isnan(ds_target)).astype(int)
    return (ds_input.fillna(0).to_numpy(), ds_target.fillna(0).to_numpy(), mask.to_numpy())


#%%

def prepare_data(
    work_dir: Path,
    data_dir: Path,   ##Changed
    year_range: tuple[int, int],
    # depths: list[float] | None = None,  ##Changed
    target_variable: str = "t_pot",
    # bathymetry_in : xr.DataArray | None = None,  ##Changed
    train_ratio = 0.7,  ##Changed
    val_ratio = 0.15   ##Changed

):
    
    #work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"
    #year_range = (1999, 2000)
    #variable = "Temperature"
    #stations = ["P22", "P23", "P24", "P25", "P26"]
    #depths = [0.5, 10.5, 50.5, 100.5]

    ## replace with saved ds
    ## ctd_filename = data_dir / "lineP_ctds/lineP_CTD_training.csv"
    start_year, end_year = year_range
    obs = load_ctd_data(Path(data_dir,Path('ctd_obs_ds_v2.nc')), start_year, end_year)
    obs=obs[[target_variable]]
    obsmask=~np.isnan(obs[target_variable])
    stations = obs['x']
    depths = obs['z']
    #obs = obs.expand_dims('channels', axis = -3)
    
    # load model data
    ds_input0, bathymetry = load_model_data(Path(data_dir,Path('griddedROMS.nc')), start_year, end_year)
    ds_input=ds_input0[[target_variable]]
    ds_target=ds_input.copy()
    ds_target[target_variable]=np.where(ds_input0['bathy'].broadcast_like(ds_input[target_variable])) 
    # apply mask based on permutations of obs sampling pattern
    perm=np.random.permutation(len(obs.time))
    perm=np.concatenate((perm,perm),axis=0)
    omask=obsmask.isel(time=perm[:len(ds_input.time)]).values
    ds_input=ds_input*omask
    
    
    # for trgt in [target_variable]:
    #     arr = ds_target[trgt].values.copy() 
    #     arr[...,model_ind_closet_to_obs] = obs[trgt].values
    #     ds_target[trgt] = (ds_target[trgt].dims, arr)

    # Add static variables
    if bathymetry is not None:  
        ds_input['bathymetry'] = (["time","z","x"],np.broadcast_to(bathymetry,ds_input[target_variable].values.shape))
    ds_input['omask']=(["time","z","x"],omask)
    ds_input['sin_yd']=ds_input0['sin_yearday'].broadcast_like(ds_input[target_variable])
    ds_input['cos_yd']=ds_input0['cos_yearday'].broadcast_like(ds_input[target_variable])
    ds_target[target_variable]=ds_input[target_variable].where(ds_input0['bathy'],ds_input[target_variable],np.nan)

    ds_input = ds_input.expand_dims('channels', axis = -3)
    ds_target= ds_target.expand_dims('channels', axis = -3)
    print('\nds_input\n',ds_input)
    print('\nds_target\n',ds_target)

    # === Split Data into train, validation, test ===
    T = ds_input.sizes["time"]
    # split ratios
    # split indices
    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)
    
    ds_input_train = ds_input.isel(time=slice(0, train_end))
    ds_input_val   = ds_input.isel(time=slice(train_end, val_end))

    ds_target_train = ds_target.isel(time=slice(0, train_end))
    ds_target_val   = ds_target.isel(time=slice(train_end, val_end))

    if train_ratio + val_ratio < 1:
        ds_input_test  = ds_input.isel(time=slice(val_end, T))
        ds_target_test  = ds_target.isel(time=slice(val_end, T))
    else:
        print('==========================================================\n'+
              'Test split ratio is zero. Test set is the same as validation set! \n' + 
              '==========================================================\n')
        ds_input_test  = ds_input_val.copy()
        ds_target_test  = ds_target_val.copy()

    # Normalization
    # Compute scale parameters from training data and apply to validation and test
    ds_input_train_norm, scale_params_in = normalize_dataset(ds_input_train)
    # Save input normalization parameters
    with open(f"{work_dir}/scale_params_in.json", "w") as f:
        json.dump(scale_params_in, f, indent=2)
    
    # Apply same normalization to validation & test inputs
    ds_input_val_norm  = apply_normalization(ds_input_val, scale_params_in)
    ds_input_test_norm = apply_normalization(ds_input_test, scale_params_in)
    
    ds_target_train_norm, scale_params_target = normalize_dataset(ds_target_train)
    # Save target normalization parameters
    with open(f"{work_dir}/scale_params_target.json", "w") as f:
        json.dump(scale_params_target, f, indent=2)
    
    # Apply same normalization to validation & test targets
    ds_target_val_norm  = apply_normalization(ds_target_val, scale_params_target)
    ds_target_test_norm = apply_normalization(ds_target_test, scale_params_target)

    print('ds_target_val_norm:\n',ds_target_val_norm)
    print('ds_target_test_norm:\n',ds_target_test_norm)

    # reshape data into graph structure, and compute target value mask
    print("\nPrepare Training:")
    train_data = reshape_to_tcsd(ds_input_train_norm, ds_target_train_norm)  ##Changed
    print("Done")
    print("\nPrepare Validation:")  
    val_data = reshape_to_tcsd(ds_input_val_norm, ds_target_val_norm)   ##Changed
    print("Done")
    print("\nPrepare Testing:")
    test_data = reshape_to_tcsd(ds_input_test_norm, ds_target_test_norm)   ##Changed
    print("Done")

    return train_data, val_data, test_data, stations, depths 


def haversine(la0,lo0,la1,lo1):
    """ haversine formula with numpy array handling
    Calculates spherical distance between points on Earth in meters
    Compares elements of (la0,lo0) with (la1,lo1)
    Shapes must be compatible with numpy array broadcasting
    args: lats and lons in decimal degrees
    returns: distance on sphere with volumetric mean Earth radius in meters
    """
    rEarth=6371*1e3 # 
    # convert to radians
    la0=np.radians(la0)
    la1=np.radians(la1)
    lo0=np.radians(lo0)
    lo1=np.radians(lo1)
    theta=2*np.arcsin(np.sqrt(np.sin((la0-la1)/2)**2+np.cos(la0)*np.cos(la1)*np.sin((lo0-lo1)/2)**2))
    d=rEarth*theta
    return d
