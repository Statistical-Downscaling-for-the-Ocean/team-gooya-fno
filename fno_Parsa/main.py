#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main/ by
@author: rlc001
"""


import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_processing import prepare_data
from train import train_model
from train import make_snapshot_data
from evaluate import evaluate_model
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

def main(output_directory, data_directory, model_directory, n_epochs, batch_size, lr , wd, reduction, early_stoppng_buffer ):

    # === Prepare Data ===
    # main_dir = "/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/FNO"
    # data_dir = "/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/data" 
    
    # main_dir = "/path/to/my/projects/line_p/"
    # data_dir = "/path/to/my/projects/line_p/data/observation"

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d-%H:%M")

    print(f'Started run with id : {formatted}')
    work_directory = output_directory / f'{formatted}'
    Path(work_directory).mkdir(parents=True, exist_ok=True)
    
    input_variable = ["Temperature"]
    target_variable = ["Temperature"]
    year_range = (1990,2025)
    # data_dir = Path(data_dir)
    # === Prepare Data ===
    train_data, val_data, test_data, stations, depths = prepare_data(
        data_dir=data_directory,
        work_dir=work_directory,
        model_dir=model_directory,
        year_range=year_range,
        # stations=["P22", "P23", "P24", "P25", "P26"],
        input_variable = input_variable,
        target_variable=target_variable,
        train_ratio=0.7, ##Changed
        val_ratio=0.15  ##Changed
    )

    # ======= model archotecture setup ==========
    width = 20
    num_layers = 4
    modes1 = 50
    modes2 = 50

    with open(Path(work_directory, "training_parameters.txt"), 'w') as f:
        f.write(
            f"year_range\t{year_range}\n" +
            f"width\t{width}\n" +
            f"num_layers\t{num_layers}\n" +
            f"modes1\t{modes1}\n"  + 
            f"modes2\t{modes2}\n" +
            f"n_epochs\t{n_epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"lr\t{lr}\n" +
            f"wd\t{wd}\n" +
            f"reduction\t{reduction}\n" +
            f"early_stoppng_buffer\t{early_stoppng_buffer}\n"  +
            f"input_variable\t{input_variable}\n"  +
            f"target_variable\t{target_variable}\n"  +
            f"data_dir\t{str(data_directory)}\n"  + 
            f"model_dir\t{str(model_directory)}\n"            
        )

    # ======= Train Model ==========
    try: ##Changed
        save_path = work_directory / f"best_model.pth"
        model, train_losses, val_losses, best_val_mse = train_model(train_data, val_data, width = width, num_layers = num_layers, modes1 = modes1, modes2 = modes2, batch_size = batch_size, n_epochs=n_epochs, lr=lr, wd= wd, reduction = reduction, early_stoppng_buffer = early_stoppng_buffer, save_path=save_path) ##Changed
    except Exception as e: ##Changed
        import shutil  ##Changed
        Path(output_directory / 'failed_cases').mkdir(parents=True, exist_ok=True)  ##Changed
        shutil.move(work_directory, output_directory / 'failed_cases')  ##Changed
        print("Terminated due to the follwoing error:\n", e)  ##Changed
        raise  #

    # === Plot learning curves ===
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.plot(np.arange(1,len(train_losses)+1), train_losses, color = 'b', label = 'train loss')
    ax.plot(np.arange(1,len(val_losses)+1), val_losses, color = 'g', label = 'validation loss')
    ax.set_title(f'Train/Val Loss - best validation score : {best_val_mse}') ###
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss :MSE - reduction : {"mean_snap"}')
    ax.legend()
    plt.show()
    plt.savefig(work_directory / f'train_val_learning_curves.png')
    plt.close()
    # === Evaluate Model ===
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    input_test, target_test, mask_test = test_data
    test_data = make_snapshot_data(input_test, target_test, mask_test)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    
    _ = evaluate_model(
        model,
        test_loader,
        target_variable=target_variable,
        stations=stations,
        depths=depths,
        work_dir=work_directory
    )
    print(f'Training finished at :\n {work_directory}')

if __name__ == "__main__":
    parser = ArgumentParser(description="Training and evaluation of downscaling Line P data using Fourier neural operator ")
    parser.add_argument("--output_directory", help="Directory to store artifacts from this script", default="/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/FNO")
    parser.add_argument("--data_directory", help="Directory for the Line P training data", default="/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/data")
    parser.add_argument("--model_directory", help="Directory for the input model", default="/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/data/NEP10k_ROMS")
    parser.add_argument("--n_epochs", help="Number of epochs", default=200, type=int)
    parser.add_argument("--batch_size", help="Number of batches", default=5, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--wd", help="weight decay", default=1e-5, type=float)
    parser.add_argument("--reduction", help="MSE reduction method", default="mean_snap")
    parser.add_argument("--early_stoppng_buffer", help="Number of epochs for early stoppng buffer (int or None)", default=None, type=int)

    args = parser.parse_args()
    main(Path(args.output_directory), Path(args.data_directory), Path(args.model_directory), args.n_epochs, args.batch_size, args.lr, args.wd , args.reduction, args.early_stoppng_buffer  )
