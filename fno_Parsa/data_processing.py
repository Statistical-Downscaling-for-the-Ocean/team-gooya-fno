#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main by 
@author: rlc001
"""

from pathlib import Path
import pandas as pd
import numpy as np
import glob
import os
import xarray as xr
import json

def load_ctd_data(data_dir, start_year, end_year, groupby_daily = False, bin_depth = True):
    """
    Load and process CTD csv files for a given year range.
    groupby_daily resamples the times as daily.
    bin_depth splits the depth into 5 meter resolution.
    Returns an xarray.Dataset with dimensions (time, channel, depth, station)).
    """
    print('==================================\n Loading CTD data ... \n ==================================')
    
    ctd_filename = data_dir / "lineP_ctds" / "lineP_CTD_training.csv"
    df_all = pd.read_csv(ctd_filename, comment="#")

    df_all["TIME"] = pd.to_datetime(df_all["TIME"], format="%Y-%m-%d %H:%M:%S")
    df_all = df_all.rename(
        columns={
            "LATITUDE": "Latitude",
            "LONGITUDE": "Longitude",
            "TEMPERATURE": "Temperature",
            "SALINITY": "Salinity",
            "OXYGEN_UMOL_KG": "Oxygen",
            "PRESSURE_BIN_CNTR": "Depth",
            "TIME": "time",
            "STATION_ID": "station",
        }
    )

    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year+1}-01-01")
    df_all = df_all[(df_all["time"] >= start_date) & (df_all["time"] < end_date)]

    if bin_depth:
       print('Binning to 5 meter resolution ...\n')
       z_edges=np.arange(0,4306,5) # 0 to 44305 inclusive 
       z_noms=(z_edges[:-1]+z_edges[1:])/2
       z_inds=np.digitize(df_all.DEPTH.values,z_edges)
       z_noms=np.append(z_noms,[None,],axis=0)
       df_all['z_nom']=[z_noms[el-1] for el in z_inds] # digitize indexes starting at 1 instead of zero
       df_all = df_all.groupby(['station', 'time', 'Latitude', 'Longitude', 'z_nom']).mean().reset_index().drop(columns=["DEPTH", 'Depth']).rename(columns={'z_nom' : 'Depth'})
    
    stations = sorted(
        df_all["station"].unique(),
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    stations.remove('P26')
    stations.append('P26')
    # Sort and get unique coords
    depths = np.sort(df_all["Depth"].unique())

    times = np.sort(df_all["time"].unique())
    longitudes = np.array([df_all[df_all['station'] == s]['Longitude'].mean() for s in stations])
    latitudes = np.array([df_all[df_all['station'] == s]['Latitude'].mean() for s in stations])
    distances = np.array([0 if ind == 0 else haversine(latitudes[ind], longitudes[ind], latitudes[ind - 1], longitudes[ind - 1]) for ind, s in enumerate(stations)])
     # # Build arrays
    variables = ["Temperature", "Salinity", "Oxygen", "Latitude", "Longitude"]
    data_dict = {var: np.full((len(times), len(stations), len(depths)), np.nan) for var in variables}

    for t_idx, t in enumerate(times):
        df_t = df_all[df_all["time"] == t]
        for s_idx, s in enumerate(stations):
            df_s = df_t[df_t["station"] == s]
            if df_s.empty:
                continue
            depth_idx = np.searchsorted(depths, df_s["Depth"])
            for var in variables:
                valid = (depth_idx >= 0) & (depth_idx < len(depths))
                data_dict[var][t_idx, s_idx, depth_idx[valid]] = df_s[var].values[valid]
    
    # Return as xarray dataset
    ds = xr.Dataset(
        {
            var: (("time", "station", "depth"), data_dict[var]) for var in variables
        },
        coords={
            "time": times,
            "station": ('station',stations),
            "lat" : ('station', latitudes),
            "lon" : ('station',longitudes),
            "distance" : ('station',distances),
            "depth": depths
        },
    )

    print(ds)

    ds["depth"].attrs["units"] = "m"
    ds["Temperature"].attrs["units"] = "deg C"
    ds["Salinity"].attrs["units"] = "PSU"
    ds["Oxygen"].attrs["units"] = "umol/kg"
    ds["Longitude"].attrs["units"] = "deg"
    ds["Latitude"].attrs["units"] = "deg"
    if groupby_daily:
        print('\n Grouping by daily ...\n')
        ds = ds.groupby(ds.time.dt.floor("D")).mean().rename({'floor' : 'time'})
        


    return ds.transpose('time', 'depth', 'station').sel(station=slice(None, None, -1))

def load_model_data(model_dir, start_year, end_year, low_res = False):
    """
    Load and process model netcdf data for a given year range.
    Also creates Bathymetry.
    Returns two xarray.Dataset with dimensions (time, channel, depth, station) and (depth, station).
    """
        
    print('==================================\n Loading model data ... \n ==================================')
    if 'NEP36_CanOE' in str(model_dir):
        ds = xr.open_dataset(model_dir / 'NEP36_along_LineP.nc')
        # ds = ds.sortby('lat')
        ds = ds.sortby('lon')
        latitudes = ds.lat.values
        longitudes = ds.lon.values
        distances = np.array([0 if ind == 0 else haversine(latitudes[ind], longitudes[ind], latitudes[ind - 1], longitudes[ind - 1]) for ind, s in enumerate(longitudes)])
        ds = ds.assign_coords(distances = ('point', distances))         ### assign distance betwen two consequetive points as a coordinate 

        ds = ds.rename({
            "point": "station",
            "salt": "Salinity",
            "temp": "Temperature",
            "O2" : 'Oxygen'})
        bathymetry =  ds['DIC'][0].where(ds['DIC'][0] == 0,1).drop(['time'])
        
    elif 'NEP10k_ROMS' in str(model_dir):
        import gsw
        if low_res:
            target_grid =  pd.read_csv('gridspecs.csv', comment="#")  ### Load the target grid for binning model data into 11km resolution. 
            tol = 0
        else:
            target_grid =  pd.read_csv('gridspecsHiRes.csv', comment="#")  ### Load the target grid for binning model data into 11km resolution.
            tol = 0.05 
        ds = xr.open_dataset(model_dir / 'nep_revised_hind_moave_all_subset_insituT.nc')
 
        ds['lon_rho'] = ds['lon_rho'] - 360 ### NEP10k has degrees East for longitude
        ds = ds.where((ds.lon_rho <= target_grid['lonedges_hi'].max()) & (ds.lon_rho >= target_grid['lonedges_lo'].min()) 
                      & (ds.lat_rho <= target_grid['latedges_hi'].max()) & (ds.lat_rho >= target_grid['latedges_lo'].min()), drop = True)   ### subselect NEP10k for line p to reduce data load
        ls = []
        
        for ind in range(len(target_grid)):
            ls.append(ds.where((ds.lon_rho <= target_grid['lonedges_hi'][ind] + tol/2) & (ds.lon_rho >= target_grid['lonedges_lo'][ind] - tol/2) 
                               & (ds.lat_rho <= target_grid['latedges_hi'][ind] + tol/2) & (ds.lat_rho >= target_grid['latedges_lo'][ind] - tol/2), drop = True).mean(('xi_rho', 'eta_rho')))  ### Bin NEP10k at the reference grid locations loaded above
        ds = xr.concat(ls, dim = 'station')
        del ls
        bathymetry_depth = ds['h_rho']
        depths = ds['zval_rho']
        ds = ds[['salt','temp']].rename({
            "salt": "Salinity",
            "temp": "Temperature",
            "ocean_time": "time"})
        ds = ds.drop('s_rho').rename({'s_rho' : 'rk' }).assign_coords(rk = depths.rk)
        z_edges=np.arange(0,4306,5) # define 5m depth coordinates for interpolation (0 to 44305 inclusive)  
        z_noms=(z_edges[:-1]+z_edges[1:])/2 
        bathymetry = xr.ones_like(bathymetry_depth).expand_dims( depth = z_noms,axis = -1)
        bathymetry = xr.concat([bathymetry.isel(station = station_id).where(bathymetry.depth <= bathymetry_depth.isel(station = station_id), 0)
            for station_id in range(len(bathymetry.station))], dim = 'station') 
        ds = xr.concat([ds.isel(station = station_id).assign_coords(rk  = depths.isel(station = station_id)).interp(rk = z_noms, kwargs={"fill_value": "extrapolate"})
                       for station_id in range(len(bathymetry.station))], dim = 'station').rename({'rk' : 'depth'})  # interpolate into 5m depth coordinates
        ds = ds.assign_coords(lon = ('station', target_grid['nom_lon'] ), lat = ('station', target_grid['nom_lat'] )).transpose('time', 'station', ...)

    ds = ds.where(bathymetry == 1)
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year+1}-01-01")
    
    ds = ds.where((ds.time >= start_date) & (ds.time < end_date), drop = True)
    return ds.transpose('time','depth','station'), bathymetry.transpose('depth','station')
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
        "Salinity": "minmax",
        "Oxygen": "zscore",
        "Bathymetry": "minmax",
        "Depth": "minmax",
        "Latitude": None,
        "Longitude": None,
        "sin_DOY" : None,
        "cos_DOY" : None,
        'mask_target' : None
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

def make_synthetic_linep(time, stations, depths) -> xr.Dataset:
   
    T = len(time)
    D = len(depths)
    S = len(stations)
    rng = np.random.default_rng(0)
    data = np.zeros((T, S, D), dtype=np.float32)

    for ti, t in enumerate(time):
        seasonal = 4.0 * np.sin(2 * np.pi * (t.dt.month - 1) / 12.0)
        for si in range(S):
            for di, depth in enumerate(depths):
                val = seasonal
                val += 0.2 * si                         
                val += np.exp(-depth / 200.0)          
                val += 0.3 * np.sin(0.1 * si * ti / max(1, S))
                val += 0.5 * rng.normal()             
                data[ti, si, di] = val + 10

    ds = xr.Dataset({"Temperature": (("time", "station", "depth"), data)}, coords={"time": time, "station": stations, "depth": depths})

    return ds

def reshape_to_tcsd(ds_input: xr.DataArray, ds_target: xr.DataArray):    ##NEW
    ds_input = xr.concat([ds_input[var] for var in list(ds_input.data_vars)], dim = 'channels')
    ds_target = xr.concat([ds_target[var] for var in list(ds_target.data_vars)], dim = 'channels')
    mask = (~np.isnan(ds_target)).astype(int)
    return (ds_input.fillna(0).to_numpy(), ds_target.fillna(0).to_numpy(), mask.to_numpy())


#%%

def prepare_data(
    work_dir: Path,
    data_dir: Path,   ##Changed
    model_dir: Path,
    year_range: tuple[int, int],
    groupby_daily = True,
    stations: list[str] | None = None,
    # depths: list[float] | None = None,  ##Changed
    input_variable: list = None, 
    target_variable: list = ["Temperature"],
    # bathymetry_in : xr.DataArray | None = None,  ##Changed
    train_ratio = 0.7,  ##Changed
    val_ratio = 0.15,   ##Changed
    low_res = False


):

    start_year, end_year = year_range
    obs = load_ctd_data(data_dir, start_year, end_year, groupby_daily=groupby_daily)
    if input_variable is None:
        input_variable = target_variable
    # Subset stations and depths
    #print(ds.station.values)
    if stations is not None: 
        obs = obs.sel(station=stations)

    #### For now to test but to be removed later ####
    # print('==========================================================\n'+
    #     'Warning! In this protocode only 4 depth points are selcted! Edit for the actual training! \n' + 
    #     '==========================================================\n')
    # depths = [2.5, 27.5, 52.5, 77.5]     ##Changed
    # obs = obs.sel(depth=depths)   ##Changed
    #################################################

    
    obs = obs[target_variable]
    # obs = obs[[target_variable]]
    obs = obs.expand_dims('channels', axis = -3)
    ### Add a code to make sure only collocated variables are kept in the obs ####
    
    # Generate synthetic line p temperature 'model' data
    # Replace this by loading model data
    # ds_input = make_synthetic_linep(ds_target['time'], ds_target['station'], ds_target['depth'])
    ds_input, bathymetry = load_model_data(model_dir, start_year, end_year, low_res=low_res)
    # ds_input = ds_input[[target_variable]].sel(time = obs.time, method = 'nearest')
    
    #### make sure model and obs have the same time range and snapshots ###
    max_time = min(ds_input.time.max() , obs.time.max())  #
    min_time = max(ds_input.time.min() , obs.time.min())
    print(f'\n Training for {min_time.values} = {max_time.values} \n')
    obs = obs.where((obs.time >= min_time)  &  (obs.time <= max_time), drop = True)   
    ds_input = ds_input.where((ds_input.time >= min_time)  &  (ds_input.time <= max_time), drop = True)   
    ds_input = ds_input[input_variable].sel(time = obs.time, method = 'nearest').expand_dims('channels', axis = -3)

    #### NEP36_CanOE has a fine horizontal resolution to which obs has to be rewritten, and depth has to be interpolated to obs ###    
    if 'NEP36_CanOE' in str(model_dir):
        ds_input = ds_input.interp(depth = obs.depth, kwargs={"fill_value": "extrapolate"})
        ds_input = ds_input.where(ds_input.lon > obs.sel(stattion = 'P23').lon, drop = True)
        obs = obs.where(obs.lon > obs.sel(stattion = 'P23').lon, drop = True)
        mask_time = obs.where(~np.isnan(obs), 0 ).sum(['depth','station'])
        obs = obs.where(mask_time != 0, drop = True)
        mask_time['time'] = ds_input['time']
        ds_input = ds_input.where(mask_time != 0, drop = True)
        del mask_time
    #### make sure obs and model have same depth ranges ###  
    if ds_input.depth.max() > obs.depth.max():
        ds_input = ds_input.where(ds_input.depth <= obs.depth.max(), drop = True)
        bathymetry = bathymetry.where(bathymetry.depth <= obs.depth.max(), drop = True)
    elif ds_input.depth.max() < obs.depth.max():
        obs = obs.where(obs.depth <= ds_input.depth.max(), drop = True)

    #### rewriting obs to model grid ###  
    # ds_target = xr.full_like(ds_input[[target_variable]], np.NaN)

    ds_target = xr.combine_by_coords([xr.full_like(ds_input[input_variable[0]], np.NaN).to_dataset(name = trgt) for  trgt in target_variable])[target_variable]
    model_ind_closet_to_obs = [np.argmin([haversine(ds_target.lat[i].values, ds_target.lon[i].values, obs.lat[j].values, obs.lon[j].values) 
                                          for i in range(len(ds_target.station))])  for j in  range(len(obs.station))] ## extracting station locations on model grid
    
    # for trgt in [target_variable]:
    for trgt in target_variable:
        arr = ds_target[trgt].values.copy() 
        arr[...,model_ind_closet_to_obs] = obs[trgt].values
        ds_target[trgt] = (ds_target[trgt].dims, arr)

    ### Add static variables ####
    if bathymetry is not None:
        if 'NEP36_CanOE' in str(model_dir):
            bathymetry = bathymetry.interp(depth = ds_input.depth)
            bathymetry = bathymetry.where(bathymetry == 0 , 1)
            ds_input = ds_input.where(bathymetry == 1)
             
        ds_input['bathymetry'] = bathymetry.broadcast_like(ds_input[input_variable[0]])

    doy = xr.DataArray(
    (obs.time.values.astype('datetime64[D]') - obs.time.values.astype('datetime64[Y]')).astype(int) + 1,
    dims=("time",),
    coords={"time": ds_input.time},
    name="DOY"
    )
    ds_input["sin_DOY"] = np.sin(doy/365.25*np.pi/180).broadcast_like(ds_input[input_variable[0]])
    ds_input["cos_DOY"] = np.cos(doy/365.25*np.pi/180).broadcast_like(ds_input[input_variable[0]])


    stations = ds_target['station']
    depths = ds_target['depth']
    # === Split Data into train, validation, test ===
    T = ds_input.sizes["time"]
    # split ratios
    # split indices
    # train_end = int(train_ratio * T)
    # val_end = int((train_ratio + val_ratio) * T)
    # ds_input_train = ds_input.isel(time=slice(0, train_end))
    # ds_input_val   = ds_input.isel(time=slice(train_end, val_end))

    # ds_target_train = ds_target.isel(time=slice(0, train_end))
    # ds_target_val   = ds_target.isel(time=slice(train_end, val_end))

    idx_train = np.sort(np.random.choice(np.arange(T), size=int(train_ratio * T), replace=False))
    mask = np.ones(np.arange(T).shape, dtype=bool)
    mask[idx_train] = False
    remaining = np.arange(T)[mask]
 
    idx_val = np.sort(np.random.choice(remaining, size=int(val_ratio * T), replace=False))

    if train_ratio + val_ratio < 1:
        idx_test = np.array([idx for idx in remaining if idx not in idx_val])

    ds_input_train = ds_input.isel(time=idx_train)
    ds_input_val   = ds_input.isel(time=idx_val)

    ds_target_train = ds_target.isel(time=idx_train)
    ds_target_val   = ds_target.isel(time=idx_val)

    if train_ratio + val_ratio < 1:
        ds_input_test  = ds_input.isel(time=idx_test)
        ds_target_test  = ds_target.isel(time=idx_test)
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
    with open(work_dir / f"scale_params_in.json", "w") as f:
        json.dump(scale_params_in, f, indent=2)

    # Apply same normalization to validation & test inputs
    ds_input_val_norm  = apply_normalization(ds_input_val, scale_params_in)
    ds_input_test_norm = apply_normalization(ds_input_test, scale_params_in)

    ds_target_train_norm, scale_params_target = normalize_dataset(ds_target_train)
    # Save target normalization parameters
    with open(work_dir/ f"scale_params_target.json", "w") as f:
        json.dump(scale_params_target, f, indent=2)

    # Apply same normalization to validation & test targets
    ds_target_val_norm  = apply_normalization(ds_target_val, scale_params_target)
    ds_target_test_norm = apply_normalization(ds_target_test, scale_params_target)

    # reshape data into graph structure, and compute target value mask
    print("\nPrepare Training:")
    train_data = reshape_to_tcsd(ds_input_train_norm, ds_target_train_norm)  ##ds_input_train_norm and ds_target_train_norm should have nans where data are missing
    print("Done")
    print("\nPrepare Validation:")
    val_data = reshape_to_tcsd(ds_input_val_norm, ds_target_val_norm)   ##Changed
    print("Done")
    print("\nPrepare Testing:")
    test_data = reshape_to_tcsd(ds_input_test_norm, ds_target_test_norm)   ##Changed
    print("Done")

    return train_data, val_data, test_data, stations, depths 



def prepare_data_for_gapfilling(
    work_dir: Path,
    data_dir: Path,   ##Changed
    model_dir: Path,
    year_range: tuple[int, int],
    groupby_daily = True,
    stations: list[str] | None = None,
    input_variable: list = None, 
    target_variable: list = ["Temperature"],
    low_res = False,
):

    start_year, end_year = year_range
    obs = load_ctd_data(data_dir, start_year, end_year, groupby_daily=groupby_daily)
    if input_variable is None:
        input_variable = target_variable
    # Subset stations and depths
    #print(ds.station.values)
    if stations is not None: 
        obs = obs.sel(station=stations)

    #### For now to test but to be removed later ####
    # print('==========================================================\n'+
    #     'Warning! In this protocode only 4 depth points are selcted! Edit for the actual training! \n' + 
    #     '==========================================================\n')
    # depths = [2.5, 27.5, 52.5, 77.5]     ##Changed
    # obs = obs.sel(depth=depths)   ##Changed
    #################################################

    
    obs = obs[target_variable]
    # obs = obs[[target_variable]]
    obs = obs.expand_dims('channels', axis = -3)
    ### Add a code to make sure only collocated variables are kept in the obs ####
    
    # Generate synthetic line p temperature 'model' data
    # Replace this by loading model data
    # ds_input = make_synthetic_linep(ds_target['time'], ds_target['station'], ds_target['depth'])
    ds_input, bathymetry = load_model_data(model_dir, start_year, end_year, low_res = low_res)
    # ds_input = ds_input[[target_variable]].sel(time = obs.time, method = 'nearest')
    
    #### make sure model and obs have the same time range and snapshots ###
    ds_input = ds_input[input_variable].expand_dims('channels', axis = -3)

    #### NEP36_CanOE has a fine horizontal resolution to which obs has to be rewritten, and depth has to be interpolated to obs ###    
    if 'NEP36_CanOE' in str(model_dir):
        ds_input = ds_input.interp(depth = obs.depth, kwargs={"fill_value": "extrapolate"})
        ds_input = ds_input.where(ds_input.lon > obs.sel(stattion = 'P23').lon, drop = True)
    #### make sure obs and model have same depth ranges ###  
    if ds_input.depth.max() > obs.depth.max():
        ds_input = ds_input.where(ds_input.depth <= obs.depth.max(), drop = True)
        bathymetry = bathymetry.where(bathymetry.depth <= obs.depth.max(), drop = True)
    elif ds_input.depth.max() < obs.depth.max():
        obs = obs.where(obs.depth <= ds_input.depth.max(), drop = True)


    ### Add static variables ####
    if bathymetry is not None:
        if 'NEP36_CanOE' in str(model_dir):
            bathymetry = bathymetry.interp(depth = ds_input.depth)
            bathymetry = bathymetry.where(bathymetry == 0 , 1)
            ds_input = ds_input.where(bathymetry == 1)
             
        ds_input['bathymetry'] = bathymetry.broadcast_like(ds_input[input_variable[0]])

    doy = xr.DataArray(
    (ds_input.time.values.astype('datetime64[D]') - ds_input.time.values.astype('datetime64[Y]')).astype(int) + 1,
    dims=("time",),
    coords={"time": ds_input.time},
    name="DOY"
    )
    ds_input["sin_DOY"] = np.sin(doy/365.25*np.pi/180).broadcast_like(ds_input[input_variable[0]])
    ds_input["cos_DOY"] = np.cos(doy/365.25*np.pi/180).broadcast_like(ds_input[input_variable[0]])

    time = ds_input['time']
    stations = ds_input['station']
    depths = ds_input['depth']
    lats = ds_input['lat']
    lons  = ds_input['lon']
    # === Split Data into train, validation, test ===
    # Normalization

    with open(work_dir / "scale_params_in.json") as f:
        scale_params_in = json.load(f)
    # Compute scale parameters from training data and apply to validation and test
  # Apply same normalization to validation & test inputs
    ds_input_norm  = apply_normalization(ds_input, scale_params_in)

    # reshape data into graph structure, and compute target value mask
    print("\nPrepare input:")
    input_data, _ ,_ = reshape_to_tcsd(ds_input_norm, ds_input_norm)  ##ds_input_train_norm and ds_target_train_norm should have nans where data are missing

    return input_data, bathymetry, time, stations, depths 


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
