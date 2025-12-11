#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main/ by
@author: rlc001
"""


import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_processing import prepare_data_for_gapfilling
from train import train_model
from train import make_snapshot_data
from evaluate import model_inference
import torch
from pathlib import Path
########### If you have torch_geometric ##############
# from torch_geometric.loader import DataLoader
######## else maske manual torch dataset ##########
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
###############################################
from datetime import datetime
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import xarray as xr
from model import FNO2d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_params(model_dir):
    params = {}
    path = glob.glob(str(model_dir / 'training_parameters.txt'))[0]
    file = open(path)
    content=file.readlines()
    for line in content:
        if '\t' in line:
            key = line.split('\t')[0]
            try:
                value = line.split('\t')[1].split('\n')[0]
            except:
                value = line.split('\t')[1]
            try:    
                params[key] = eval(value)
            except:
                if key == 'ensemble_list':
                    ls = []
                    for item in value.split('[')[1].split(']')[0].split(' '):
                        try:
                            ls.append(eval(item))
                        except:
                            pass
                    params[key] = ls
                else:
                    params[key] = value
    return params

def predict(work_directory, year_range = None, low_res = False):

    model_params = extract_params(work_directory)
    
    input_variable = model_params['input_variable']
    target_variable = model_params['target_variable']
    data_directory = Path(model_params['data_dir'])
    model_directory =  Path(model_params['model_dir'])
    if year_range is None:
        year_range = model_params['year_range']
    
    if low_res != model_params['low_res']:
            D = 850
            if model_params['low_res']:
                reschange = 'LowtoHigh'
                S = 38 
            else:
                reschange = 'HightoLow'
                S = 114 
    else:
        reschange = 'SameRes'
    # data_dir = Path(data_dir)
    # === Prepare Data ===
    input_data, bathymetry ,times, stations, depths = prepare_data_for_gapfilling(
        data_dir=data_directory,
        work_dir=work_directory,
        model_dir=model_directory,
        year_range=year_range,
        # stations=["P22", "P23", "P24", "P25", "P26"],
        input_variable = input_variable,
        target_variable=target_variable,
        low_res = low_res
    )


    # ======= Train Model ==========

    save_path = work_directory / f"best_model.pth"
    modes1 = model_params['modes1']
    modes2 = model_params['modes2']

    if modes1 is None: ##NEW
        modes1  = D  
    if modes2 is None:  ##NEW
        modes2 = np.floor(S/2) + 1 

    model =  FNO2d(input_data.shape[1], len(model_params['target_variable']) , model_params['width'] , modes1, modes2 , num_layers  = model_params['num_layers'])  ##New
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.to(device)

    test_data = make_snapshot_data(input_data, input_data, None)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    
    output = model_inference(
        model,
        test_loader,
        target_variable=target_variable,
        stations=stations,
        depths=depths,
        work_dir=work_directory
    )

    ds = xr.Dataset(
        {
            var: (("time", "depth", "station"), output[:,ind]) for ind, var in enumerate(target_variable)
        },
        coords={
            "time": times.values,
            "station": ('station',stations.values),
            "lat" : ('station', stations.lat.values),
            "lon" : ('station',stations.lon.values),
            "depth": depths.values
        },
    )

    ds = ds.where(bathymetry == 1)
    model_name = str(model_directory).split('/')[-1]
    ds.to_netcdf(work_directory / f'gapfilled_lineP_{model_name}_grid_{year_range[0]}_{year_range[1]}_{reschange}.nc')

    print(f'Finished at :\n {work_directory}')

if __name__ == "__main__":
    parser = ArgumentParser(description="Inferencen for downscaling Line P data using Fourier neural operator ")
    parser.add_argument("--work_directory", help="Directory where model is saved")
    parser.add_argument("--year_range", help="inference period", type=tuple, default=None)
    parser.add_argument("--low_res", help="predict on lower resolution?", type=bool, default=False)

    args = parser.parse_args()
    predict(Path(args.work_directory) , args.year_range,  args.low_res)
