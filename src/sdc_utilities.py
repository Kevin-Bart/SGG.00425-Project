# Copyright 2023 GRID-Geneva. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import os
import glob
import datetime
# import sys
import rasterio
# import warnings
# import re
from osgeo import gdal
import numpy as np
import pandas as pd
import xarray as xr

from IPython.display import HTML
from math import floor
# from datetime import datetime
# from numpy.lib.stride_tricks import as_strided
# from pyproj import Transformer
# from itertools import product as iterprod

import ast
from odc.stac import stac_load

from odc.algo import mask_cleanup
from datacube.testutils.io import rio_slurp
from datacube.storage import measurement_paths
from datacube.utils import geometry
from copy import copy
# from utils.data_cube_utilities.dc_utilities import clear_attrs

gdal.UseExceptions()

# SCL class colors
scl_colors = {
  0: [0, 0, 0], #// No Data (Missing data) - black  
  1: [255, 0, 0], #// Saturated or defective pixel - red 
  2: [47, 47, 47], #// Topographic casted shadows ("Dark features/Shadows" for data before 2022-01-25) - very dark grey
  3: [100, 50, 0], #// Cloud shadows - dark brown
  4: [0, 160, 0], #// Vegetation - green
  5: [255, 230, 90], #// Not-vegetated - dark yellow
  6: [0, 0, 255], #// Water (dark and bright) - blue
  7: [128, 128, 128], #// Unclassified - dark grey
  8: [192, 192, 192], #// Cloud medium probability - grey
  9: [255, 255, 255], #// Cloud high probability - white
  10: [100, 200, 255], #// Thin cirrus - very bright blue
  11: [255, 150, 255], #// Snow or ice - very bright pink
}


def fix_crs(ds):
    '''
    This fixes the 'grid_mapping' attribute that causes wrong crs association when openening exported netcdfs
    '''
    # Fix CRS problems for export
    for var in ds.data_vars:
        if 'grid_mapping' in ds[var].encoding:
            del ds[var].encoding['grid_mapping']
    
    # ds['mask'].attrs['grid_mapping'] = 'spatial_ref'
    for var in ds.data_vars:
        ds[var].attrs['grid_mapping'] = 'spatial_ref'

    # ds['mask'].attrs['grid_mapping'] = 'spatial_ref'  # Ensure grid_mapping in attrs
    # ds['mask'].encoding.pop('grid_mapping', None)
        
    return ds
    
def load_product_ts(catalog=None, product=None, **kwargs):

    assert 'measurements' in kwargs.keys(), \
           "\n! <measurements> is required !"
    assert 'QA_PIXEL' in kwargs['measurements'] or 'SCL' in kwargs['measurements'], \
           "\n! <measurements> must contains 'QA_PIXEL' or 'SCL' band !"
    assert 'resolution' in kwargs.keys() and 'output_crs' in kwargs.keys(), \
       "\n! <resolution> and <output_crs> are required !"

    qa_name = None
    if 'SCL' in kwargs['measurements']:
        qa_name = 'SCL'
    elif 'QA_PIXEL' in kwargs['measurements']:
        qa_name = 'QA_PIXEL'
    else:
        qa_name = None
        
    # Extract optional parameters with default values if not provided
    time = kwargs.get('time')
    longitude = kwargs.get('longitude')
    latitude = kwargs.get('latitude')
    measurements = kwargs.get('measurements')
    output_crs = kwargs.get('output_crs')
    resolution = kwargs.get('resolution', [None, None])
    rename = kwargs.get('rename', False)
    alias_names = kwargs.get('alias_names')
    chunks = kwargs.get('chunks', None)
    scale_offset = kwargs.get('scale_offset', None)   # Check if qa_name is QA_PIXEL

    # Build query with conditional checks for optional params
    datetime_range = f"{time[0]}/{time[1]}" if time else None
    bbox = (longitude[0], latitude[0], longitude[1], latitude[1]) if longitude and latitude else None

    query = catalog.search(
        collections=[product],
        datetime=datetime_range,
        limit=100,
        bbox=bbox
    )
    items = list(query.items())

    # Load identified items
    lazy_ds = stac_load(
        items,
        lon=longitude,
        lat=latitude,
        bands=measurements,
        crs=output_crs,
        resolution=resolution[1] if resolution else None,
        chunks=chunks,
    )

    # Fix CRS problems for export
    lazy_ds = fix_crs(lazy_ds)
        
    # Scale if neccessary
    if scale_offset is not None:
        
        if qa_name == 'QA_PIXEL':
            lazy_ds[qa_name].attrs['units'] = 'bit_index'
            lazy_ds[qa_name].attrs['flags_definition'] = []
            lazy_ds[qa_name] = lazy_ds[qa_name].astype(np.int64)
            exclude_bands = ['QA_PIXEL','mask']
            if product == 'landsat_ot_c2_l2':
                # Load Landsat 8 scale factors and offsets
                scaling_factors = pd.read_csv('data/lsc2_scale_table_89.csv')
                # print(scaling_factors)
            elif product == 'landsat_etm_c2_l2' or product == 'landsat_tm_c2_l2':
                # Load Landsat 8 scale factors and offsets
                scaling_factors = pd.read_csv('data/lsc2_scale_table_457.csv')
                # print(scaling_factors)
            else:
                warnings.warn("Landsat product not identified.")
                
            
            
            # Apply scale and offset to each band
            for band, scale, offset in zip(scaling_factors['Band_Designation'],
                                           scaling_factors['Multiplicative_Scale_Factor'],
                                           scaling_factors['Additive_Offset']):
                if band in lazy_ds.data_vars and band not in exclude_bands:
                    print(f"Applying scaling {scale} and offset to {band}")
                    lazy_ds[band] = (lazy_ds[band] * scale) + offset
    
            print("Applied Landsat scaling factors and offset successful.") 
    
        elif qa_name == 'SCL':
            exclude_bands = ['SCL']
            if product == 's2_l2':
                scaling_factor = 1/10000
                # get the alias names. Apply scaling factor only to the regular reflectance bands
                 # is tirs a band? #
                refl_al = ['blue','green','red','red_edge_1','red_edge_2','red_edge_3','nir_1',
                           'nir_2','swir_16','swir_222'] #alias names
                refl_me = [ 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
                # Apply scale and offset to each band
                for band in refl_me:
                        print(f"Applying scaling to {band}")
                        lazy_ds[band] = (lazy_ds[band] * scaling_factor) 
    
                print("Applied Sentinel 2 scaling factors successful.") 
    
    # # Add coords and crs for qa_name; export does not include those for some reason
    # if qa_name is not None:
    #     # get dims and coords from first measurement that is no == qa_name
    #     for _meas in measurements:
    #         if _meas != qa_name:
    #             # lazy_ds[qa_name].dims = lazy_ds[_meas].dims
    #             # lazy_ds[qa_name].coords = lazy_ds[_meas].coords
    #             # lazy_ds[qa_name].attrs['crs'] = lazy_ds[_meas].attrs.get('crs', None)
    #             # break  # Stop after the first match
                
    #             # Ensure the dimensions match by renaming as necessary
    #             lazy_ds[qa_name] = lazy_ds[qa_name].swap_dims({dim: lazy_ds[_meas].dims[i] for i, dim in enumerate(lazy_ds[qa_name].dims)})
                
    #             # Set coordinates and attributes
    #             lazy_ds[qa_name] = lazy_ds[qa_name].assign_coords(lazy_ds[_meas].coords)
    #             # lazy_ds[qa_name].attrs['crs'] = lazy_ds[_meas].attrs.get('crs', None)
    #             break  # Stop after the first match
     
    
    # Rename variables if needed
    if rename and alias_names:
        _olds = list(lazy_ds.data_vars.keys())
        for r in range(len(alias_names)):
            if _olds[r] != alias_names[r]:  # only non-identical, otherwise error
                lazy_ds = lazy_ds.rename({_olds[r]: alias_names[r]})        

    return lazy_ds


def get_alias_band(product,alias, measurements_file):
    # create list for each row and save as new row entry
    # overview file of datasets
    df = pd.read_csv(measurements_file)
    dfs = df.loc[df['product']==product]
    # measurements_aliases = ['QA_PIXEL', 'blue', 'green', 'red', 'nir', 'swir_1', 'swir_2']
    bands_to_keep = []
    bands_alias = []
    for r in range(len(dfs.index)):
        # r['measurement']
        _alias = dfs.iloc[r,:]['aliases']
        _alias = ast.literal_eval(_alias)
        _meas = dfs.iloc[r,:]['measurement']
        _alias.append(_meas)
        # check if any of the "measurements_aliases" is inside _alias
        _match = set(alias).intersection(_alias)
        if len(_match) > 0:
        # if any(map(lambda v: v in measurements_aliases, _alias)):
            # actual band name
            bands_to_keep.append(_meas)
            # keep also the alias name, at least return this as info
            bands_alias.append(_match.pop())
    
    print(f"Product: {product}")
    # print(f"Bands found in product: {bands_alias}")
    # print(f"Band names real       : {bands_to_keep}")
    
    
    df_out = pd.DataFrame({'measurement': bands_to_keep, 'alias': bands_alias})
    print(df_out)

    return bands_to_keep, bands_alias


def create_scl_clean_mask(scl, valid_cats = [4, 5, 6, 7, 11]):
    """
    Description:
      Create a clean mask from a list of valid categories,
    Input:
      scl (xarray) - SCL from dc_preproc product (generated with sen2cor)
    Args:
      scl: xarray data array to extract clean categories from.
      valid_cats: array of ints representing what category should be considered valid.
      * category selected by default
      ###################################
      # SCL categories:                 #
      #   0 - no data                   #
      #   1 - saturated or defective    #
      #   2 - dark area pixels          #
      #   3 - cloud_shadows             #
      #   4 * vegetation                #
      #   5 * not vegetated             #
      #   6 * water                     #
      #   7 * unclassified              #
      #   8 - cloud medium probability  #
      #   9 - cloud high probability    #
      #  10 - thin cirrus               #
      #  11 * snow                      #
      ###################################
    Output:
      clean_mask (boolean numpy array)
    """

    return xr.apply_ufunc(np.isin, scl, valid_cats).values


def create_mask_from_bits(ds, bit_positions=[]):
    '''
    This creates a mask (0 for valid, 1 for invalid) from the QA_PIXEL layer in Landsat.
    '''

    # Make sure the datatype is int64 first:
    ds['QA_PIXEL'] = ds['QA_PIXEL'].astype(np.int64)
    
    # Initialize the mask array with 0 (indicating valid data by default)
    valid_mask = xr.DataArray(
        data=np.zeros(ds['QA_PIXEL'].shape, dtype=int),  # Initialize with zeros
        dims=ds['QA_PIXEL'].dims,                        # ('time', 'y', 'x')
        coords=ds['QA_PIXEL'].coords                     # Copy coordinates
    )
    
    # Loop through each bit position and update the mask
    for bit in bit_positions:
        # Create a boolean mask for where the bit is set, and use logical OR to update the valid_mask
        bit_mask = (ds['QA_PIXEL'] & (1 << bit)) != 0
        valid_mask = valid_mask | bit_mask.astype(int)

    # Assign the mask to the dataset (0 for valid, 1 for invalid)
    ds['mask'] = valid_mask
    
    # Fix CRS problem
    # ds['mask'].attrs['grid_mapping'] = 'spatial_ref'  # Ensure grid_mapping in attrs
    # ds['mask'].encoding.pop('grid_mapping', None)
    ds = fix_crs(ds)

    return ds
    
# def create_mask_from_bits(ds, bit_positions=[]):
#     '''
#     This creates a mask (0 for valid, 1 for invalid) from the QA_PIXEL layer in Landsat.
#     '''

#     # Make sure the datatype is int64 first:
#     ds['QA_PIXEL'] = ds['QA_PIXEL'].astype(np.int64)
#     # Initialize the mask array to 0 (indicating valid data by default)
#     # valid_mask = np.zeros(ds['QA_PIXEL'].shape, dtype=int)
#     valid_mask = xr.DataArray(
#         data=np.zeros(ds['QA_PIXEL'].shape),  # Replace with desired values or initialization
#         dims=ds['QA_PIXEL'].dims,             # ('time', 'y', 'x')
#         coords=ds['QA_PIXEL'].coords          # Copy coordinates from an existing layer
#     )
    
#     # Function to update the mask for each bit position in the list
#     def apply_bit_mask(bit_position):
#         # Create a mask that sets invalid values (1) where the specific bit is set
#         valid_mask[(ds['QA_PIXEL'] & (1 << bit_position)) != 0] = 1
    
#     # Apply mask for each bit position in the list
#     for bit in bit_positions:
#         apply_bit_mask(bit)
    
#     # Convert the mask to an xarray DataArray and add it to the dataset
#     # valid_mask_da = xr.DataArray(valid_mask, dims=ds['QA_PIXEL'].dims, coords=ds['QA_PIXEL'].coords)
#     # ds['mask'] = valid_mask_da
#     ds['mask'] = valid_mask
    
#     return ds


def create_mask_from_values(ds, invalid_values=[]):
    '''
    This creates a mask (0 for valid, 1 for invalid) from the SCL layer in Sentinel 2.
    '''
    mask = (ds['SCL'] < 0) | ds['SCL'].isin(invalid_values)    
    
    # Convert the mask to an xarray DataArray and add it to the dataset
    valid_mask_da = xr.DataArray(mask, dims=ds['SCL'].dims, coords=ds['SCL'].coords)
    ds['mask'] = valid_mask_da

    # Fix CRS
    ds = fix_crs(ds)
    
    return ds


# def ls_qa_clean(dc_qa, valid_bits = [1, 2, 4]):
#     """
#     Description:
#       create a clean mask of a Landsat Collection 1 dataset using pixel_qa band and a list of valid bits
#     Input:
#       dc_qa: pixel_qa band of a Landast Collection 1 xarray.DataArray
#     Args:
#       valid_bits: array of ints representing which bit should be considered as valid (default: clear, water, snow)
#       #############################################
#       # BITS : CATEGORIES                         #
#       #    0 : Fill                               #
#       #    1 : Clear                              #
#       #    2 : Water                              #
#       #    3 : Cloud shadow                       #
#       #    4 : Snow                               #
#       #    5 : Cloud                              #
#       #   10 : Terrain occlusion (Landsat 8 only) #
#       #############################################
#     Output:
#       clean_mask (boolean numpy array)
#     """

#     # Check submitted input
#     if str(type(dc_qa)) != "<class 'xarray.core.dataarray.DataArray'>":
#         sys.exit("SCRIPT INTERRUPTED: dc_qa should be an xarray.DataArray")
#     if dc_qa.name != "pixel_qa":
#         sys.exit("SCRIPT INTERRUPTED: dc_qa name  should be pixel_qa")

#     # List and count all dc_qa unique values
#     dc_qas, dc_cnt = unik_count(dc_qa.values)
#     # Return bit encoding
#     bit_len = bit_length(max(dc_qas))

#     # First keep only low confidence cloud (and cirrus)
#     ok_qas = []
#     ko_qas = []

#     if bit_len == 8: # Landsat 5 and 7
#         for v in sorted(dc_qas):
#             b = str(bin(v))[2:].zfill(bit_len)[::-1]
#             if b[6] == '1' and b[7] == '0':
#                 ok_qas.append(v)
#             else:
#                 ko_qas.append(v)

#     if bit_len >= 10: # Landsat 8 (>= as sometimes pixel_qa become 11 bit !!!)
#         for v in sorted(dc_qas):
#             b = str(bin(v))[2:].zfill(bit_len)[::-1]
#             if b[6] == '1' and b[7] == '0' and b[8] == '1' and b[9] == '0':
#                 ok_qas.append(v)
#             else:
#                 ko_qas.append(v)

#     # Second keep only valid_bits
#     data_qas = []
#     nodata_qas = []
#     for v in sorted(ok_qas):
#         b = str(bin(v))[2:].zfill(bit_len)[::-1]
#         for c in valid_bits:
#             if b[c] == '1':
#                 data_qas.append(v)
#                 break

#     return xr.apply_ufunc(np.isin, dc_qa, data_qas, dask = 'allowed').values

def lsc2_loadcleanscale(dc, products, valid_cats = [], mask_filters = [], scale = True, **kwargs):
    """
    Description:
      Create a clean and scaled single Landsat dataset (multi-product or not) using cleaning "autor's recommended ways" lsc2_qa_clean
      Scene without any data removed, sorted by ascending time 
    Input:
      dc:           datacube.api.core.Datacube
                    The Datacube instance to load data with.
    Args:
      products:     list of products
      valid_cats:   array of integers representing what category should be considered valid
      mask_filters: list of any mask_filters from mask_cleanup function
      any other default argument from dc.load (time, lon, lat, measurements, output_crs, resolution, resampling,...)
      
    Output:
      cleaned and scaled (even if it can be disabled) dataset and clean_mask sorted by ascending time
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 23.02.2023)
    """
    
    # Check submitted input
    # Convert product string into list
    if isinstance(products, str):
        products = products.split()
        
    assert 'measurements' in kwargs.keys(), \
           "\n! <measurements> is required !"
    assert 'QA_PIXEL' in kwargs['measurements'], \
           "\n! <measurements> must contains 'QA_PIXEL' band !"
    
    # Load and combine dataset
    ds_out = None
    for product in products:
        # load product dataset
        ds_tmp = dc.load(product = product, **kwargs)        
        
        if len(ds_tmp.variables) == 0: continue # skip the current iteration if empty

        # clean product dataset
        if len(valid_cats) == 0: valid_cats = [5,6,7]
        clean_mask_tmp = lsc2_qa_clean(ds_tmp.QA_PIXEL, valid_cats)
        ds_tmp = ds_tmp.drop_vars('QA_PIXEL')
        
        # morphologicaly modify the mask if required
        if len(mask_filters) > 0:
            clean_mask_tmp = mask_cleanup(clean_mask_tmp, mask_filters)
        
        # apply mask
        ds_tmp = ds_tmp.where(clean_mask_tmp)
        
        # remove time without any data
        ds_tmp = ds_tmp.dropna('time', how='all')
        
        # scale if required (default)
        if scale:
            ds_tmp = lsc2_ds_scale(ds_tmp, product)
        
        # initiate or append to dataset to return
        if ds_out is None:
            ds_out = ds_tmp.copy(deep=True)
        else:
            ds_out = xr.concat([ds_out, ds_tmp], dim = 'time')
        del ds_tmp

    if ds_out is not None:
        # sort dataset by ascending time
        ds_out = ds_out.sortby('time')
        return (ds_out, ~np.isnan(ds_out[list(ds_out.var())[0]].values))
    else:
        return (0, 0)

def lsc2_loadcleanscale_nodc(catalog,  products, valid_cats = [], mask_filters = [], scale = True, **kwargs):
    # measurements,
    """
    Description:
      Create a clean and scaled single Landsat dataset (multi-product or not) using cleaning "autor's recommended ways" lsc2_qa_clean
      Scene without any data removed, sorted by ascending time 
    Input:
    # ----- removed dc dependency -----
      dc:           datacube.api.core.Datacube
                    The Datacube instance to load data with.
    # ----- included catalog dependency -----                
      catalog:      dask catalog
    # ----- included measurements dependency -----                
      catalog:      dask catalog
      
    Args:
      products:     list of products
      valid_cats:   array of integers representing what category should be considered valid
      mask_filters: list of any mask_filters from mask_cleanup function
      any other default argument from dc.load (time, lon, lat, measurements, output_crs, resolution, resampling,...)
      
    Output:
      cleaned and scaled (even if it can be disabled) dataset and clean_mask sorted by ascending time
    Authors:
      Bruno Chatenoux (UNEP/GRID-Geneva, 23.02.2023)
    """
    
    # Check submitted input
    # Convert product string into list
    if isinstance(products, str):
        products = products.split()
        
    assert 'measurements' in kwargs.keys(), \
           "\n! <measurements> is required !"
    assert 'QA_PIXEL' in kwargs['measurements'], \
           "\n! <measurements> must contains 'QA_PIXEL' band !"

    measurements = kwargs.get('measurements')
    # Load and combine dataset
    ds_out = None
    for product in products:
        # load product dataset
        # change this to no use a dc object
        # ds_tmp = dc.load(product = product, **kwargs)  
        # ---- go through the catalog loading process
        
        ds_tmp = load_product_ts(catalog=catalog,
                                 product=product,
                                 **kwargs
                                )
        
        if len(ds_tmp.variables) == 0: continue # skip the current iteration if empty

        # clean product dataset
        if len(valid_cats) == 0: valid_cats = [5,6,7]
        # units required:
        if 'QA_PIXEL' in ds_tmp:
            # Check if 'units' attribute exists for 'QA_PIXEL' variable
            if 'units' not in ds_tmp['QA_PIXEL'].attrs:
                # Assign a default value to the 'units' attribute
                ds_tmp['QA_PIXEL'].attrs['units'] = 'bit_index'  
        else:
            print("QA_PIXEL variable not found in the dataset.")
        clean_mask_tmp = lsc2_qa_clean(ds_tmp.QA_PIXEL, valid_cats)
        ds_tmp = ds_tmp.drop_vars('QA_PIXEL')
        
        # morphologicaly modify the mask if required
        if len(mask_filters) > 0:
            clean_mask_tmp = mask_cleanup(clean_mask_tmp, mask_filters)
        
        # apply mask
        ds_tmp = ds_tmp.where(clean_mask_tmp)
        
        # remove time without any data
        ds_tmp = ds_tmp.dropna('time', how='all')
        
        # scale if required (default)
        if scale:
            ds_tmp = lsc2_ds_scale(ds_tmp, product)
        
        # initiate or append to dataset to return
        if ds_out is None:
            ds_out = ds_tmp.copy(deep=True)
        else:
            ds_out = xr.concat([ds_out, ds_tmp], dim = 'time')
        del ds_tmp

    if ds_out is not None:
        # sort dataset by ascending time
        ds_out = ds_out.sortby('time')
        return (ds_out, ~np.isnan(ds_out[list(ds_out.var())[0]].values))
    else:
        return (0, 0)
        
# def load_multi_clean(dc, products, valid_cats = [], mask_filters = [], **kwargs):
#     """
#     Description:
#       Create a clean dataset (multi-product or not) using cleaning "autor's recommended ways"
#       - lsc2_qa_clean
#       - create_slc_clean_mask
#       Scene without any data removed, sorted by ascending time 
#     Input:
#       dc:           datacube.api.core.Datacube
#                     The Datacube instance to load data with.
#     Args:
#       products:     list of products
#       valid_cats:   array of integers representing what category should be considered valid
#       mask_filters: list of any mask_filters from mask_cleanup function
#       any other default argument from dc.load (time, lon, lat, output_crs, resolution, resampling,...)
      
#     Output:
#       cleaned dataset and clean_mask sorted by ascending time
#     Authors:
#       Bruno Chatenoux (UNEP/GRID-Geneva, 23.02.2023)
#     """
    
#     # Check submitted input
#     # Convert product string into list
#     if isinstance(products, str):
#         products = products.split()
        
#     load_multi_clean.__globals__.update(kwargs)
#     assert any(item in measurements for item in ['QA_PIXEL', 'SCL']), \
#         "\n! <measurements> must contains either 'QA_PIXEL' (Landsat), or 'SCL' (Sentinel 2) !"
    
#     # Load and combine dataset
#     ds_out = None
#     for product in products:
#         # load product dataset
#         ds_tmp = dc.load(product = product, **kwargs)        
        
#         if len(ds_tmp.variables) == 0: continue # skip the current iteration if empty

#         # clean product dataset
#         if 'QA_PIXEL' in measurements:
#             if len(valid_cats) == 0: valid_cats = [5,6,7]
#             clean_mask_tmp = lsc2_qa_clean(ds_tmp.QA_PIXEL, valid_cats)
#             ds_tmp = ds_tmp.drop_vars('QA_PIXEL')
#         elif 'SCL' in measurements:
#             if len(valid_cats) == 0: valid_cats = [4, 5, 6, 7, 11]
#             clean_mask_tmp = create_scl_clean_mask(ds_tmp.SCL, valid_cats)
#             clean_mask_tmp = xr.DataArray(clean_mask_tmp, coords = ds_tmp.coords, dims = ds_tmp.dims)
#             ds_tmp = ds_tmp.drop_vars('SCL')
        
#         # modify the mask if required
#         if len(mask_filters) > 0:
#             clean_mask_tmp = mask_cleanup(clean_mask_tmp, mask_filters)
        
#         # apply mask
#         ds_tmp = ds_tmp.where(clean_mask_tmp)
        
#         # remove time without any data
#         ds_tmp = ds_tmp.dropna('time', how='all')
        
#         # initiate or append to dataset to return
#         if ds_out is None:
#             ds_out = ds_tmp.copy(deep=True)
#         else:
#             ds_out = xr.concat([ds_out, ds_tmp], dim = 'time')
#         del ds_tmp

#     if ds_out is not None:
#         # sort dataset by ascending time
#         ds_out = ds_out.sortby('time')
#         return (ds_out, ~np.isnan(ds_out[measurements[0]].values))
#     else:
#         return (0, 0)

# def load_multi_clean(dc, products, measurements, valid_cats = [], **kwargs):
#     """
#     Description:
#       Create a clean dataset (multi-product or not) using cleaning "autor's recommended ways"
#       - ls_qa_clean
#       - create_slc_clean_mask
#       Scene without any data removed, sorted by ascending time 
#     Input:
#       dc:           datacube.api.core.Datacube
#                     The Datacube instance to load data with.
#     Args:
#       products:     list of products
#       valid_cats:   array of ints representing what category should be considered valid
#                     * category selected by default
#       # SENTINEL 2 ################################
#       #   0 - no data                             #
#       #   1 - saturated or defective              #
#       #   2 - dark area pixels                    #
#       #   3 - cloud_shadows                       #
#       #   4 * vegetation                          #
#       #   5 * not vegetated                       #
#       #   6 * water                               #
#       #   7 * unclassified                        #
#       #   8 - cloud medium probability            #
#       #   9 - cloud high probability              #
#       #  10 - thin cirrus                         #
#       #  11 * snow                                #
#       #############################################
#       # LANDSAT 5, 7 and 8 ########################
#       #    0 : Fill                               #
#       #    1 * Clear                              #
#       #    2 * Water                              #
#       #    3 : Cloud shadow                       #
#       #    4 * Snow                               #
#       #    5 : Cloud                              #
#       #   10 : Terrain occlusion (Landsat 8 only) #
#       #############################################
#       any other default argument from dc.load (time, lon, lat, output_crs, resolution, resampling,...)
      
#     Output:
#       cleaned dataset and clean_mask sorted by ascending time
#     Authors:
#       Bruno Chatenoux (UNEP/GRID-Geneva, 15.03.2022)
#     """
    
#     # Check submitted input
#     # Convert product string into list
#     if isinstance(products, str):
#         products = products.split()
        
#     # Get common measurements
#     common_measurements = []
#     measurement_list = dc.list_measurements(with_pandas=False)
#     for product in products:
#         measurements_for_product = filter(lambda x: x['product'] == product, measurement_list)
#         common_measurements.append(set(map(lambda x: x['name'], measurements_for_product)))
#     common_measurements = list(set.intersection(*map(set, common_measurements)))
#     assert len(common_measurements) > 0, \
#            '! No common measurements found'
    
#     # Check requested measurements are in common measurements
#     assert all([item in common_measurements for item in measurements]), \
#            f"""
#            All requested measures are not available for each product
#            Only {common_measurements} are available
#            """
    
#     # Add quality measurement for Landsat or Sentinel 2 products
#     # using the first product as they shouldn't be mixed
#     if products[0][:2] == 'ls' and 'pixel_qa' not in measurements:
#         measurements.append('pixel_qa')
#     elif products[0][:2] == 's2' and 'slc' not in measurements:
#         measurements.append('slc')
    
#     # Load and combine dataset
#     ds_out = None
#     for product in products:
#         # load product dataset
#         ds_tmp = dc.load(product = product, measurements = measurements, **kwargs)        
        
#         if len(ds_tmp.variables) == 0: continue # skip the current iteration if empty

#         # clean product dataset
#         if products[0][:2] == 'ls':
#             if len(valid_cats) == 0: valid_cats = [1, 2, 4]
#             clean_mask_tmp = ls_qa_clean(ds_tmp.pixel_qa, valid_cats)
#         elif products[0][:2] == 's2':
#             if len(valid_cats) == 0: valid_cats = [4, 5, 6, 7, 11]
#             clean_mask_tmp = create_slc_clean_mask(ds_tmp.slc, valid_cats)
#         ds_tmp = ds_tmp.where(clean_mask_tmp)
        
#         # remove time without any data
#         ds_tmp = ds_tmp.dropna('time', how='all')
        
#         # initiate or append to dataset to return
#         if ds_out is None:
#             ds_out = ds_tmp.copy(deep=True)
#         else:
#             ds_out = xr.concat([ds_out, ds_tmp], dim = 'time')
#         del ds_tmp

#     if ds_out is not None:
#         # sort dataset by ascending time
#         ds_out = ds_out.sortby('time')
#         return (ds_out, ~np.isnan(ds_out[measurements[0]].values))
#     else:
#         return (0, 0)


# # source: https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
# def tile_array(a, x0, x1, x2):
#     t, r, c = a.shape                                    # number of rows/columns
#     ts, rs, cs = a.strides                                # row/column strides 
#     x = as_strided(a, (t, x0, r, x1, c, x2), (ts, 0, rs, 0, cs, 0)) # view a as larger 4D array
#     return x.reshape(t*x0, r*x1, c*x2)                      # create new 2D array

# def updown_sample(ds_l, ds_s, resampl):
#     """
#     Description:
#       Up or down sample a "large" resolution xarray.Dataset (so far Landsat products) and a "small" resolution
#       xarray.Dataset (so far Sentinel 2 product) and combine them into a single xarray.Dataset.
#       "large" resolution must be a multiple of "small" resolution and geographical extent must be adequate.
#       Xarray.Dataset need to be cleaned as mask band will be removed from the output
#       To enforce this requirement usage of load_lss2_clean function (without the resampl option) is
#       highly recommended.

#     Args:
#       ds_l:         'large' resolution xarray.Dataset
#       ds_s:         'small' resolution xarray.Dataset
#       resampl:      'up' to upsample
#                     'down_mean' to downsample using mean values
#                     'down_median' to downsample using median values
      
#     Output:
#       Upsampled and combined dataset and clean_mask sorted by ascending time.
#     Authors:
#       Bruno Chatenoux (UNEP/GRID-Geneva, 11.12.2019)
#     """
    
#     # check resampl options
#     resampl_opts = ['up', 'down_mean', 'down_median']
#     assert (resampl in resampl_opts) or (resampl == ''), \
#            '\nif used, resample option must be %s' % resampl_opts
    
#     # check ds ratio
#     ratiox = len(ds_s.longitude.values) / len(ds_l.longitude.values)
#     ratioy = len(ds_s.latitude.values) / len(ds_l.latitude.values)
#     assert (ratiox == 3), \
#            '\nthe ratio of the number of columns should be 3 (Landsat/Sentinel 2 only so far) !'
#     assert (ratioy == 3), \
#            '\nthe ratio of the number of rows should be 3 (Landsat/Seentinel 2 only so far) !'

#     # check ds resolutions
#     resx_l = (ds_l.longitude.values.max() - ds_l.longitude.values.min()) / (len(ds_l.longitude.values) - 1)
#     resy_l = (ds_l.latitude.values.max() - ds_l.latitude.values.min()) / (len(ds_l.latitude.values) - 1)
#     resx_s = (ds_s.longitude.values.max() - ds_s.longitude.values.min()) / (len(ds_s.longitude.values) - 1)
#     resy_s = (ds_s.latitude.values.max() - ds_s.latitude.values.min()) / (len(ds_s.latitude.values) - 1)
#     # in reason of proper float storage issue, compare resolution with a 0.1% accuracy
#     assert ((abs(resx_s - resx_l / ratiox) / resx_s * 100) < 0.1), \
#            '\nthe column resolution is not a mutiple of %i !' % (ratiox)
#     assert ((abs(resy_s - resy_l / ratioy) / resy_s * 100) < 0.1), \
#            '\nthe row resolution is not a mutiple of %i !' % (ratioy)
    
#     # check spacing of ds top left pixel center with a 0.1%
#     assert ((abs(ds_l.longitude.values.min() - ds_s.longitude.values.min()) - resx_s) < resx_s * 0.001), \
#            '\nthe longitudinal extent of both dataset do not overlay properly !' + \
#            '\nuse load_lss2_clean function to fix this issue'
#     assert ((abs(ds_l.latitude.values.min() - ds_s.latitude.values.min()) - resy_s) < resy_s * 0.001), \
#            '\nthe latitudinal extent of both dataset do not overlay properly !' + \
#            '\nuse load_lss2_clean function to fix this issue'
    
#     # check vars (without mask band as they will no be combined)
#     vars_l = [ele for ele in sorted(list(ds_l.data_vars)) if ele not in ['pixel_qa', 'slc']]
#     vars_s = [ele for ele in sorted(list(ds_s.data_vars)) if ele not in ['pixel_qa', 'slc']]
#     assert (vars_l == vars_s), \
#            '\nmeasurements in dataset are not identical'
    
#     # upsample "large" dataset (using temporary array)
#     for index, var in enumerate(vars_l):
#         if resampl == 'up':
#             arr_l = tile_array(ds_l[var].values, 1, int(ratiox), int(ratioy))
#             da_l = xr.DataArray(arr_l, dims=['time', 'latitude', 'longitude'])
#             da_l = da_l.assign_coords(time = ds_l.time,
#                                         latitude = ds_s.latitude,
#                                         longitude = ds_s.longitude)
#             # combine s and l
#             da = xr.concat([ds_s[var], da_l], dim = 'time')
#         elif resampl[:5] == 'down_':
#             # source: https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy/42463491#42463491
#             # 4x faster than skimage way (who has an issue with median function in the case of large stdev !)
#             t, lat, lon = ds_s[var].values.shape
#             nlat = lat // ratiox
#             nlon = lon // ratioy
#             if resampl == 'down_median':
#                 arr_s = np.nanmedian(ds_s[var].values[:1*t, :int(nlat*ratioy), :int(nlon*ratiox)]. \
#                         reshape(1, t, int(nlat), int(ratioy), int(nlon), int(ratiox)), axis=(0, 3, 5))
#             elif resampl == 'down_mean':
#                 arr_s = np.nanmean(ds_s[var].values[:1*t, :int(nlat*ratioy), :int(nlon*ratiox)]. \
#                         reshape(1, t, int(nlat), int(ratioy), int(nlon), int(ratiox)), axis=(0, 3, 5))
#             da_s = xr.DataArray(arr_s, dims=['time', 'latitude', 'longitude'])
#             da_s = da_s.assign_coords(time = ds_s.time,
#                                       latitude = ds_l.latitude,
#                                       longitude = ds_l.longitude)
#             # combine l and s
#             da = xr.concat([ds_l[var], da_s], dim = 'time')
        
#         if index == 0:   
#             ds = da.to_dataset(name = var)
#         else:
#             ds = ds.merge(da.to_dataset(name = var))

#     # Sort dataset by ascending time
#     ds = ds.sortby('time')
    
#     return ds

# def load_lss2_clean(dc, products, time, lon, lat, measurements,
#                    resampl = '', valid_cats = [[],[]]):
#     """
#     Description:
#       Create a clean dataset mixing Landsat and Sentinel 2 products (respectively with prefixs 'ls' and
#       's2')
#       and using cleaning "autor's recommended ways":
#       - ls_qa_clean
#       - create_slc_clean_mask
#       Sorted by ascending time
#       If resample option is activated ('up' or 'down_mean', 'down_median') up/downsampling is performed
#       and products output combined into a single 'lss2' prefix
#       This function works as load_multi_clean function, but with a mix of Landsat and Sentinel 2 products
#       the resampl option was added (to optionally combine products output)

#     Input:
#       dc:           datacube.api.core.Datacube
#                     The Datacube instance to load data with.
#     Args:
#       products:     list of products
#       time:         pair (list) of minimum and maximum date
#       lon:          pair (list) of minimum and maximum longitude
#       lat:          pair (list) of minimum and maximum longitude
#       measurements: list of measurements (without mask band, landsat and Sentinel 2 products prefix
#                     shouls be 'ls or 's2)
#       resampl:      (OPTIONAL) Up/Downsample ('up', 'down_mean', 'down_median' ) products and combine
#                     their output
#       valid_cats:   (OPTIONAL) list of list of ints representing what category should be considered valid
#                     first Landsat categories, then Sentinel 2 categories
#                     * meand category by default
#       # SENTINEL 2 ################################
#       #   0 - no data                             #
#       #   1 - saturated or defective              #
#       #   2 - dark area pixels                    #
#       #   3 - cloud_shadows                       #
#       #   4 * vegetation                          #
#       #   5 * not vegetated                       #
#       #   6 * water                               #
#       #   7 * unclassified                        #
#       #   8 - cloud medium probability            #
#       #   9 - cloud high probability              #
#       #  10 - thin cirrus                         #
#       #  11 * snow                                #
#       #############################################
#       # LANDSAT 5, 7 and 8 ########################
#       #    0 : Fill                               #
#       #    1 * Clear                              #
#       #    2 * Water                              #
#       #    3 : Cloud shadow                       #
#       #    4 * Snow                               #
#       #    5 : Cloud                              #
#       #   10 : Terrain occlusion (Landsat 8 only) #
#       #############################################
#     Output:
#       cleaned dataset and clean_mask sorted by ascending time stored in dictionnaries,
#       if no up/downsampling is performed dictionnaries contains the two Landsat and Sentinel 2 output
#       products
#     Authors:
#       Bruno Chatenoux (UNEP/GRID-Geneva, 11.12.2019)
#     """
    
#     # intersect measurements with common measurements
#     measurement_list = dc.list_measurements(with_pandas=False)
#     for index, product in enumerate(products):
#         measurements_for_product = filter(lambda x: x['product'] == product, measurement_list)
#         valid_measurements_name_array = set(map(lambda x: x['name'], measurements_for_product))
#         if index == 0:
#             common_measurements = sorted(valid_measurements_name_array)
#         else:
#             common_measurements = sorted(set(common_measurements).intersection(valid_measurements_name_array))
#     measurements = sorted(set(measurements).intersection(common_measurements))
    
#     # dictionary sensor -> mask band (Higher resolution first !)
#     dict_sensmask = {'ls':'pixel_qa',
#                      's2': 'slc'}
    
#     resampl_opts = ['up', 'down_mean', 'down_median']
    
#     # check mix Landsat and Sentinel 2
#     sensors = []
#     for product in products:
#         if product[:2] not in sensors:
#             sensors.append(product[:2])
#     assert (sorted(set(sensors)) == sorted(set(dict_sensmask.keys()))), \
#            '\nA mix of Landsat and Sentinel 2 products is required !\nYou should use load_multi_clean function'
    
#     assert (len(valid_cats) == 2), \
#            '\nvalid_cats argument must be a list of list (read the doc for more details)'
    
#     assert (resampl in resampl_opts) or (resampl == ''), \
#            '\nif used, resample option must be %s' % resampl_opts
    
#     dict_dsc = {}
#     dict_cm = {}
    
#     # Process first Landsat and then Sentinel 2 (based on dict_sensmask order)
#     for index, sensor in enumerate(dict_sensmask.keys()):
#         # fix Sentinel 2 geographical extent based on Landsat dataset
#         if index == 1:
#             resx = (dsc.longitude.values.max() - dsc.longitude.values.min()) / len(dsc.longitude.values)
#             resy = (dsc.latitude.values.max() - dsc.latitude.values.min()) / len(dsc.latitude.values)
#             lon = (dsc.longitude.values.min() - resx / 3, dsc.longitude.values.max() + resx / 3)
#             lat = (dsc.latitude.values.min() - resy / 3, dsc.latitude.values.max() + resy / 3)
        
#         dsc, cm = load_multi_clean(dc = dc,
#                                   products = [prod for prod in products if prod[:2] == sensor] ,
#                                   time = time,
#                                   lon = lon,
#                                   lat = lat,
#                                   measurements = measurements + [dict_sensmask[sensor]], # append mask band
#                                   valid_cats = valid_cats[index])
#         dict_dsc[sensor] = dsc
#         dict_cm[sensor] = cm
    
#     if resampl in resampl_opts :
#         dsc = updown_sample(dict_dsc['ls'], dict_dsc['s2'], resampl)
#         dict_dsc = {}
#         dict_cm = {}
#         dict_dsc['lss2'] = dsc
#         dict_cm['lss2'] = ~np.isnan(dsc[measurements[0]].values)
    
#     return dict_dsc, dict_cm


def _get_transform_from_xr(dataset):
    """Create a geotransform from an xarray dataset.
    """

    cols = len(dataset.longitude)
    rows = len(dataset.latitude)
    pixelWidth = abs(dataset.longitude[-1] - dataset.longitude[0]) / (cols - 1)
    pixelHeight = abs(dataset.latitude[-1] - dataset.latitude[0]) / (rows - 1)

    from rasterio.transform import from_bounds
    geotransform = from_bounds(dataset.longitude[0] - pixelWidth / 2, dataset.latitude[-1] - pixelHeight / 2,
                               dataset.longitude[-1] + pixelWidth / 2, dataset.latitude[0] + pixelHeight / 2,
                               cols, rows)
    return geotransform


def write_geotiff_from_xr(tif_path, dataset, bands = None, no_data = -9999,
                          crs = None, compr = ''):
    """
    Write a geotiff from an xarray dataset
    Modified for SDC:
    - fixed pixel shift bug
    - original band name added to band numbers
    - compression option added

    Args:
        tif_path: path for the tif to be written to.
        dataset: xarray dataset
        bands: (OPTIONAL) list of strings representing the bands in the order
        they should be written, or all <dataset> bands by default.
        no_data: (OPTIONAL) nodata value for the dataset (-9999 by default)
        crs: (OPTIONAL) requested crs (in the case the info is not available in <dataset>
        compr: (OPTIONAL) compression option (None by default), could be e.g. 'DEFLATE' or 'LZW'

    """
    # Check CRS information is correctly provided
    try:
        ds_crs = dataset.crs
        if crs is None:
            crs = ds_crs
        elif crs != ds_crs:
            crs = None # as a direct assert returns an error and switch to except
    except:
        assert crs is not None, \
               '<dataset> do not contains crs attribute, you have to fill <crs>!'
    # assert outside of try as it returns an error and switch to except
    assert crs is not None, \
           '<crs> differ from <dataset> crs, simply keep <crs> empty!'
    
    # Check band information
    if bands is None:
        bands = list(dataset.data_vars)
    assert isinstance(bands, list), "Bands must a list of strings"
    assert len(bands) > 0 and isinstance(bands[0], str), "You must supply at least one band."
    
    # Create the geotiff
    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=dataset.dims['latitude'],
            width=dataset.dims['longitude'],
            count=len(bands),
            dtype=dataset[bands[0]].dtype,
            crs=crs,
            transform=_get_transform_from_xr(dataset),
            nodata=no_data,
            compress=compr) as dst:
        for index, band in enumerate(bands):
            dst.write(dataset[band].values, index + 1)
        dst.close()
    
    # set band names
    ds = gdal.Open(tif_path, gdal.GA_Update)
    for index, band in enumerate(bands):
        rb = ds.GetRasterBand(index + 1)
        rb.SetDescription(band)
    del ds
    

def new_get_query_metadata(dc, product, quick = False):
    """
    Gets a descriptor based on a request.

    Args:
        dc: The Datacube instance to load data with.
        product (string): The name of the product associated with the desired dataset.
        quick (boolean): Attempt to quickly get metadata from a small dataset, and process
                         the full dataset if not possible. tile_count will not be evaluated
                         with this option.

    Returns:
        scene_metadata (dict): Dictionary containing a variety of data that can later be
                               accessed.
    """
    estim_dict = {}
    if dc.list_products().loc[product,].default_crs is None:
        estim_dict['output_crs'] = 'epsg:4326'
        estim_dict['resolution'] = (-0.01, 0.01)
    
    todo = True
    if quick:
        limit = 10
        ds = dc.load(product, measurements = [], limit = limit, **estim_dict)
        if len(ds.time) == limit:
            todo = False
            tile_count = 'not calculated with quick option'
    if todo:
        ds = dc.load(product, measurements = [], **estim_dict)
        tile_count = ds.time.size
    
    if len(set(ds.dims).intersection(['x', 'y'])) >= 1:
        ds = ds.rename({'x': 'longitude', 'y': 'latitude'})
    
    if estim_dict:
        print("\nThe product don't have a defined CRS and resolution, then geographical extents are estimated in epsg:4326 !\n")
        resx, resy = None, None
        crs = None
        minx = min(ds.longitude.values)
        miny = min(ds.latitude.values)
        maxx = max(ds.longitude.values)
        maxy = max(ds.latitude.values)
    else:
        resx = (max(ds.longitude.values) - min(ds.longitude.values)) / (len(ds.longitude) - 1)
        resy = (max(ds.latitude.values) - min(ds.latitude.values)) / (len(ds.latitude) - 1)
        crs = ds.crs
        minx = min(ds.longitude.values) - resx / 2
        miny = min(ds.latitude.values) - resy / 2
        maxx = max(ds.longitude.values) + resx / 2
        maxy = max(ds.latitude.values) + resy / 2
    
    
    return {'lon_extents': (minx, maxx),
            'lat_extents': (miny, maxy),
            'lon_res': resx,
            'lat_res': resy,
            'crs': crs,
            'time_extents': (ds.time[0].values.astype('M8[ms]').tolist(),
                             ds.time[-1].values.astype('M8[ms]').tolist()),
            'tile_count': tile_count}
    
# def summarize_products_extents(dc, products, **kwargs):
#     """
#     Returns the maximum extent (in space and time) of a given list of products.
#     Args:
#         dc: The Datacube instance to load data with
#         products (list): List of products to get metadata from.

#     Returns:
#         scene_metadata (dict): Dictionary of min and max extents.
#     """
#     miny, maxy = 1E27, -1E27
#     minx, maxx = 1E27, -1E27
#     start_date, end_date = datetime.strptime('2050-12-31', '%Y-%m-%d'), datetime.strptime('1970-01-01', '%Y-%m-%d')
#     for product in products:
#         mt = new_get_query_metadata(dc, product, **kwargs)
#         miny = mt['lat_extents'][0] if mt['lat_extents'][0] < miny else miny
#         maxy = mt['lat_extents'][1] if mt['lat_extents'][1] > maxy else maxy
#         minx = mt['lon_extents'][0] if mt['lon_extents'][0] < minx else minx
#         maxx = mt['lon_extents'][1] if mt['lon_extents'][1] > maxx else maxx
#         start_date = mt['time_extents'][0] if mt['time_extents'][0] < start_date else start_date
#         end_date = mt['time_extents'][1] if mt['time_extents'][1] > end_date else end_date
    
#     return {'lon_extents': (minx, maxx),
#             'lat_extents': (miny, maxy),
#             'time_extents': (start_date, end_date)}


# def get_products_attributes(dc, qry, cols = ['name', 'crs', 'resolution']):
#     """
#     Description:
#       Get products attributes using a query (WITHOUT "", e.g. products['name'].str.startswith('SPOT'))
#     Input:
#       dc:           datacube.api.core.Datacube
#                     The Datacube instance to load data with.
#     Args:
#       qry:          query string, e.g.:
#                     "products['name'].str.startswith('SPOT')"
#                     "products['name'].str.match('^SPOT.*$')" should give the same result as startswith example
#                     "products['name'].str.match('^SPOT.*_PAN_scene$')"
                    
#       cols:         (OPTIONAL) list of column names to get (you can view the column available by running 'dc.list_products().columns')
#     Output:
#       pandas.Dataframe
#     Authors:
#       Bruno Chatenoux (UNEP/GRID-Geneva, 5.11.2020)
#     """
#     products = dc.list_products()
#     prod_df = products[eval(qry)][cols].reset_index().drop(['id'], axis=1)
#     prod_df['measurements'] = prod_df.apply(lambda row: sorted(map(lambda x: x['name'],
#                                                                    filter(lambda x: x['product'] == row['name'],
#                                                                           dc.list_measurements(with_pandas=False)))), axis=1)
#     return(prod_df)

# def time_list(ds):
#     time_list = []
#     for i in range(len(ds.time)):
#         time_list.append(i)
#     return time_list

# # source: https://stackoverflow.com/questions/57856010/automatically-optimizing-pandas-dtypes
# def optimize_types(dataframe):
#     """
#     The function takes in a dataframe and returns the same dataframe with optimized data types
    
#     :param dataframe: the dataframe to optimize
#     :return: the dataframe with the optimized types.
#     """
#     np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
#                np.uint8 ,np.uint16, np.uint32, np.uint64]
#     np_types = [np_type.__name__ for np_type in np_types]
#     type_df = pd.DataFrame(data=np_types, columns=['class_type'])

#     type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
#     type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
#     type_df['range'] = type_df['max_value'] - type_df['min_value']
#     type_df.sort_values(by='range', inplace=True)
#     for col in dataframe.loc[:, dataframe.dtypes <= np.integer]:
#         col_min = dataframe[col].min()
#         col_max = dataframe[col].max()
#         temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
#         optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
#         dataframe[col] = dataframe[col].astype(optimized_class)
#     return dataframe

# def df_point_append_values(df, df_lon, df_lat, df_crs, ds, pts_shift = 0):
#     """
#     The function takes a csv dataframe, a xarray.Dataset, the names of the longitude and latitude
#     columns in the csv dataframe, the crs of the csv dataframe, and the shift value (in degrees) to be
#     added to the csv dataframe longitude and latitude columns (in csv dataframe coordinates units)
    
#     :param df: the dataframe to which you want to append the xarray values
#     :param df_lon: the name of the longitude column in your csv file
#     :param df_lat: the name of the latitude column in your csv file
#     :param df_crs: the coordinate reference system of the csv file
#     :param ds: the xarray dataset to get values from
#     :param pts_shift: the number of pixels to shift the csv points up and right , defaults to 0 (optional,
#     this option is usefull when for example points location correspond to the exact corner of the xarray
#     dataset (positive value are appropriate to lower-left corner)
#     :return: A dataframe with the same number of rows as the input dataframe, and one column for each of
#     the variables in the xarray.Dataset.
#     """
#     # get the real bbox of the dataset (as by default extent is given to the center of corner pixels)
#     ds_min_lon = float(ds.longitude.min())
#     ds_max_lon = float(ds.longitude.max())
#     ds_min_lat = float(ds.latitude.min())
#     ds_max_lat = float(ds.latitude.max())

#     # get the resolution of a pixel
#     resx = (ds_max_lon - ds_min_lon) / (len(ds.longitude.values) - 1)
#     resy = (ds_max_lat - ds_min_lat) / (len(ds.latitude.values) - 1)
#     # extend by half a pixel
#     ds_min_lon = ds_min_lon - (resx / 2)
#     ds_max_lon = ds_max_lon + (resx / 2)
#     ds_min_lat = ds_min_lat - (resy / 2)
#     ds_max_lat = ds_max_lat + (resy / 2)
    
#     ds_crs = int(ds.crs.split(':')[1])

#     # reproject real bbox corners from ds to csv CRS
#     # source: https://hatarilabs.com/ih-en/how-to-translate-coordinate-systems-for-xy-point-data-tables-with-python-pandas-and-pyproj
#     transformer = Transformer.from_crs(f"epsg:{ds_crs}", f"epsg:{df_crs}",always_xy=True)
#     corners = list(iterprod([ds_min_lon, ds_max_lon], [ds_min_lat, ds_max_lat]))
#     trans_corners = np.array(list(transformer.itransform(corners)))   
    
#     # clip the csv dataframe with reprojected bbox
#     df = df[(df[df_lon] + pts_shift >= np.min(trans_corners[:, 0])) &
#             (df[df_lon] + pts_shift <= np.max(trans_corners[:, 0])) &
#             (df[df_lat] + pts_shift>= np.min(trans_corners[:, 1])) &
#             (df[df_lat] + pts_shift <= np.max(trans_corners[:, 1]))]
    
#     # reproject csv dataframe coordinates to ds CRS
#     transformer = Transformer.from_crs(f"epsg:{df_crs}", f"epsg:{ds_crs}",always_xy=True)
#     points = list(zip(df[df_lon],df[df_lat]))
#     trans_coords = np.array(list(transformer.itransform(points)))
    
#     # append trans_coords as get_x and get_y (coordinated to be used to get pixel values in xarray.Dataset)
#     pd.options.mode.chained_assignment = None # fix for "A value is trying to be set on a copy of a slice from a DataFrame."
#     df['get_x'] = trans_coords[:,0]
#     df['get_y'] = trans_coords[:,1]
    
#     # Get values of xarray.Dataset on points coordinates and append to csv dataframe
#     # get
#     ds_pts = ds.sel(longitude = xr.DataArray(df.get_x, dims=["point"]),
#                     latitude = xr.DataArray(df.get_y, dims=["point"]),
#                     method="nearest")
#     df_pts = ds_pts.to_dataframe().drop(['latitude', 'longitude'], axis = 1)
#     if 'time' in df_pts.columns:
#         df_pts = df_pts.drop(['time'], axis = 1)
        
#     # deal with duplicated column names
#     for c in list(df_pts.columns):
#         if c in list(df.columns):
#             df_pts.rename(columns={c: f"{c}_joined"}, inplace=True)
        
#     # append
#     df = df.drop(['get_x', 'get_y'], axis = 1)
#     df = df.join(df_pts)
    
#     return df


# def indices_ts_stats(ds, sm_dict, idces_dict, stats, nanpercs = None, verbose = False):
#     """
#     Given a dataset, a dictionary of "seasons", a dictionary of indices, a list of statistics, and
#     optionnaly a list of percentiles the function returns a dataset with the statistics of the indices
#     per season.
    
#     :param ds: the dataset to be analyzed
#     :param sm_dict: a dictionary with seasons names as keys and lists of months as values.
#      'all' values will use the full dataset.
#     :param idces_dict: a dictionnary with the indices names as keys and related functions as values.
#      The dataset name in the functions need to be 'ds'.
#     :param stats: a list of statistical functions to be applied to the data
#      numpy by default, 'range' can also be used
#     :param nanpercs: list of percentiles to calculate (OPTIONAL, required if np.nanpercentile in
#      <stats>)
#     :param verbose: Print processing description (OPTIONAL, False by default)
    
#     :return: A dataset with the requested statistics per season and indice
#     """
#     if np.nanpercentile in stats:
#         assert 'nanpercs' in locals(), \
#                "!!!  <nanpercs> is required with 'np.nanpercentile' !!!"
    
#     first = True
#     for s, m in sm_dict.items():
#         if verbose: print(f"  └{s}")
#         if m == 'all':
#             ds_s = ds
#         else:
#             ds_s = ds.sel(time=ds.time.dt.month.isin(m))
#         if (not isinstance(ds_s, xr.Dataset)) or (len(ds_s.time) == 0):
#             continue
            
#         for idces,form in idces_dict.items():
#             if verbose: print(f"   └{idces}")
#             da_idces = eval(form.replace('ds.', 'ds_s.'))
#             da_idces = da_idces.where(np.isfinite(da_idces)) # replace +-Inf by nan
            
#             for i in range(0, len(stats)):
#                 if stats[i] == 'range':
#                     da_stat = xr.DataArray(np.max(a = da_idces, axis = 0) - np.min(a = da_idces, axis = 0),
#                                            dims = ['latitude', 'longitude'])
#                     stat_name = stats[i]
#                     if verbose: print(f"     └{stat_name}")
#                     ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
#                                                     latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
#                     del da_stat
#                     if first:
#                         first = False
#                         ds_stats = ds_stat
#                     else:
#                         ds_stats = ds_stats.merge(ds_stat)
#                     del ds_stat
#                 elif stats[i].__name__ == 'nanpercentile':
#                     for pc in nanpercs:
#                         da_stat = xr.DataArray(stats[i](a = da_idces, q = pc, axis = 0),
#                                                dims = ['latitude', 'longitude'])
#                         stat_name = f"{stats[i].__name__}{pc:02d}"
#                         if verbose: print(f"     └{stat_name}")
#                         ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
#                                                         latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
#                         del da_stat
#                         if first:
#                             first = False
#                             ds_stats = ds_stat
#                         else:
#                             ds_stats = ds_stats.merge(ds_stat)
#                     del ds_stat
#                 else:
#                     da_stat = xr.DataArray(stats[i](a = da_idces, axis = 0),
#                                            dims = ['latitude', 'longitude'])
#                     stat_name = stats[i].__name__ 
#                     if verbose: print(f"     └{stat_name}")
#                     ds_stat = da_stat.assign_coords(longitude = da_idces.longitude.values,
#                                                     latitude = da_idces.latitude.values).to_dataset(name = f'{s}_{idces}_{stat_name}')
#                     del da_stat
#                     if first:
#                         first = False
#                         ds_stats = ds_stat
#                     else:
#                         ds_stats = ds_stats.merge(ds_stat)
#                     del ds_stat
#             del da_idces
#         del ds_s
#     return ds_stats


def get_native_epsg_and_res(dc, product, measur):
    """
    Description:
      Returns the native epsg code and resolution of a given measurement (of a given product)
    Args:
      dc:      The Datacube instance to load data with.
      product: product to get information from
      measur:  measurement to get information from  
    Output:
      native dataset epsg code, x and abs(y) resolutions
    Source:
      https://gist.github.com/robbibt/51a85978afd9e94b5da0762d31ba3d7c
    """
    # Check if measurement exists and search for a potential alias
    mtd = dc.list_measurements().loc[product]
    if not measur in mtd['name']:
        no_measur = True
        for row in mtd.itertuples():
            if measur in row.aliases:
                measur = row.name
                no_measur = False
                break
        if no_measur:
            sys.exit('SCRIPT INTERRUPTED: provided measurements does not exists')
    
    ds = dc.find_datasets(product=product)[0]
    band_path = measurement_paths(ds)[measur]
    raster_meta = rio_slurp(band_path)
    return(raster_meta[1].crs.to_epsg(), raster_meta[1].transform[0], abs(raster_meta[1].transform[4]))

def ll_to_utm(longitude: float, latitude: float) -> int:
    """
    Convert geographic longitude and latitude to UTM EPSG code.
    Args:
        longitude (float): Longitude in decimal degrees.
        latitude (float): Latitude in decimal degrees.
    Returns:
        int: UTM EPSG code.
    """
    zone_number = floor((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number
    return epsg_code

def ll_to_utm(longitude, latitude):
    zone_number = floor((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number
    return epsg_code

def reproj_pt(x, y, from_epsg, to_epsg):
    """
    Description:
      Returns reprojected point cordinates 
    Args:
      x:          x coodinate of the point to reproject
      y:          y coodinate of the point to reproject
      from_epsg:  EPSG code of the initial CRS
      to_epsg:    EPSG code of the destination CRS
    Output:
      reprojected x and y resolutions
    """
    p = geometry.point(x, y, crs=geometry.CRS(f"EPSG:{from_epsg}"))
    pg = p.to_crs(geometry.CRS(f"EPSG:{to_epsg}"))
    return pg.xy[0][0], pg.xy[1][0]

def reproj_res(dd_lon, dd_lat, from_res_x, from_res_y, from_epsg, to_epsg):
    """
    Description:
      Returns reprojected coordinate of a point in a given coordinate in decimal degrees
      typically used with `dc.load` parameter longitude/latitude (in EPSG:4326) 
    Args:
      dd_lon:     point longitude in decimal degress (EPSG:4326)
      dd_lat:     point latitude in decimal degrees (EPSG:4326)
      from_res_x: x resolution in from CRS defined by <from_epsg>
      from_res_y: y resolution in from CRS defined by <from_epsg>
      from_epsg:  EPSG code of the given resolution
      to_epsg:    EPSG code to convert resolutions to
    Output:
      reprojected x and y resolutions
    """
    if from_res_y < 0:
        from_res_y = -from_res_y
    # reproject dd point in from crs
    from_x, from_y = reproj_pt(dd_lon, dd_lat, 4326, from_epsg)
    # reproject to to crs ll and ur corners of a virtual pixel
    to_x_ll, to_y_ll = reproj_pt(from_x - from_res_x / 2,
                                 from_y - from_res_y / 2, from_epsg, to_epsg)
    to_x_ur, to_y_ur = reproj_pt(from_x + from_res_x / 2,
                                 from_y + from_res_y / 2, from_epsg, to_epsg)
    return to_x_ur - to_x_ll, to_y_ur - to_y_ll


def show_bit_flags(dc, prod, measur):
    """
    Description:
      Return a Pandas DataFrame bit value(s) attributed to flags
    Args:
      dc:      The Datacube instance to load data with.
      prod:   product
      measur: measurement (requires a flag definition)
    """
    m_dict = {}
    df = None
    measurements = dc.list_measurements().loc[prod]
    flags_dict = measurements.loc[measurements.name == measur, 'flags_definition'][measur]
    assert isinstance(flags_dict, dict), \
           f"SCRIPT INTERRUPTED: no flag_definition found for measurement '{measur}'"
    for k, v in flags_dict.items():
        # print(k, v['bits'])
        if isinstance(v['bits'], list):
            bit = min(v['bits'])
        else:
            bit = v['bits']
        m_dict[f"{k}: {v['bits']}"] =  bit
    for k in dict(sorted(m_dict.items(), key=lambda x:x[1])).keys():
        insert_row = {'bit(s)': k.split(':')[1],
                     'flag': k.split(':')[0],}
        df = pd.concat([df, pd.DataFrame([insert_row])], ignore_index=True)
    return df

# Return unique values and count
def unik_count(vals):
    bc = vals.flatten()
    # make sure the object is int64
    bc = bc.astype(np.int64)
    bc = np.bincount(bc)
    unik = np.nonzero(bc)[0]
    cnt = bc[unik] * 100
    return (unik, cnt)

# Return bit length
def bit_length(da):
    v_max = 0
    for v in da.flags_definition.values():
        v = v['bits']
        if isinstance(v, list):
            v = max(v)
        if v > v_max:
            v_max = v
    return v_max + 1

def lsc2_qa_clean(da_qa, valid_bits = None, invalid_bits = None,
                  rm_high_conf = True, rm_mid_conf = True, force_snow = False):
    """
    Description:
      Creates a clean mask of a Landsat Collection 2 dataset using a given band
      and a list of valid bits.
      
      The function was written to supports loading the standard SDC products:
      - landsat_tm_c2_l2
      - landsat_etm_c2_l2
      - landsat_ot_c2_l2
      And tested with the following measurements:
      - QA_PIXEL
      - SR_CLOUD_QA (Landsat tm and etm only)
      
      Default <valid_bits> depends of the selected quality measurement:
      - QA_PIXEL: [5, 6, 7] (Snow, Clear, Water)
      - SR_CLOUD_QA: [0, 4, 5] (dark_dense_vegetation, snow, water)
      Your can use `show_bit_flags(<product>, <measurement>)`
      
      You can visualize the flag attributed to bit(s) using the `show_bit_flags` function
      Or have a look at official documentation:
      - https://www.usgs.gov/media/files/landsat-4-7-collection-2-level-2-science-product-guide
      - https://www.usgs.gov/media/files/landsat-8-collection-2-level-2-science-product-guide
      
    Args:
      da_qa:        quality band of a Landsat Collection 2 xarray.DataArray
      valid_bits:   (OPTIONAL) list of valid bits (see description above)
      invalid_bits: (OPTIONAL) list of bits considered an invalid (as often a pixel can
                    be flagged with several categories
      rm_high_conf: (OPTIONAL, only used with QA_PIXEL measurement and True by default)
                    Removes high confidence cloud, cloud shadow and cirrus to prioritize
                    snow
      rm_mid_conf:  (OPTIONAL, only used with QA_PIXEL measurement and True by default)
                    Removes mid confidence cloud to prioritize snow
      force_snow:   (OPTIONAL, False by default) consider any pixel with snow high confidence
                    as data even if it also has a high (or medium) confidence level for cloud,
                    cloud shadow or cirrus
      
    Output:
      clean_mask (boolean numpy array)
    """
    
    # Check submitted input
    assert isinstance(da_qa, xr.DataArray), \
            "SCRIPT INTERRUPTED: da_qa should be an xarray.DataArray"
    assert da_qa.units == 'bit_index', \
        f"SCRIPT INTERRUPTED: no flag_definition found for measurement '{da_qa.name}'"
    
    if da_qa.name != 'QA_PIXEL' and rm_high_conf is not None \
       and rm_mid_conf is not None:
        warnings.warn("<rm_high_conf> and <rm_mid_conf> not used")
    
    # List and count all da_qa unique values
    ok_qas, da_cnt = unik_count(da_qa.values)
    ok_qas = list(ok_qas)
    # Return bit encoding    
    bit_len = bit_length(da_qa)
    
    if da_qa.name == 'QA_PIXEL':
        # List snow forced qas
        if force_snow:
            snow_qas = []
            for v in sorted(ok_qas):
                b = str(bin(v))[2:].zfill(bit_len)[::-1]
                if b[5] == '1' and b[12:14] == '11':
                    snow_qas.append(v)
        
        if valid_bits is None:
            valid_bits = [5, 6, 7]
        
        # Get rid of high confidence cloud, cloud shadow and cirrus
        if rm_high_conf:
            for v in sorted(ok_qas):
                b = str(bin(v))[2:].zfill(bit_len)[::-1]
                if b[8:10] == '11' or b[10:12] == '11' or b[14:16] == '11':
                    ok_qas.remove(v)
        
        # Get rid of mid confidence cloud
        if rm_mid_conf:
            for v in sorted(ok_qas):
                b = str(bin(v))[2:].zfill(bit_len)[::-1]
                if b[8:10] == '01':
                    ok_qas.remove(v)
            
        # Append snow forced qas
        if force_snow:
            ok_qas = list(set(ok_qas + snow_qas))            
            
    elif da_qa.name == 'SR_CLOUD_QA':
        if valid_bits is None:
            valid_bits = [0, 4, 5]
    else:
        if valid_bits is None:
            sys.exit('SCRIPT INTERRUPTED: you need to define <valid_bits>, ' \
                     'use `show_bit_flags(<product>, <measurement>)` to make sure ' \
                     'you define proper bits measurement and values')
    
    # Keep only valid_bits
    data_qas = []
    for v in sorted(ok_qas):
        b = str(bin(v))[2:].zfill(bit_len)[::-1]
        for c in valid_bits:
            if b[c] == '1':
                data_qas.append(v)
    data_qas = sorted(list(set(data_qas)))
    
    # Remove invalid_bits
    if invalid_bits is not None:
        for v in sorted(data_qas):
            b = str(bin(v))[2:].zfill(bit_len)[::-1]
            for c in invalid_bits:
                if b[c] == '1':
                    data_qas.remove(v)
                    break
        data_qas = sorted(list(set(data_qas)))      
    
    bool_arr = xr.apply_ufunc(np.isin, da_qa, data_qas, dask = 'allowed').values
    return xr.DataArray(bool_arr, coords=da_qa.coords, dims=da_qa.dims, attrs=da_qa.attrs)

def decod_bit(v, da_qa):
    bits_dict = {'0': 'No',
                 '1': 'Yes',
                '00': 'No confidence level set',
                '01': 'Mid/Reserved',
                '10': 'Low',
                '11': 'High'}
    df_decoded = pd.DataFrame(columns=['bits', 'zo', 'flag', 'Value', 'tmp'])
    bit_len = bit_length(da_qa)
    b = str(bin(v))[2:].zfill(bit_len)[::-1]
    for k, itm in da_qa.flags_definition.items():        
        if isinstance(itm['bits'], list):
            zo = b[itm['bits'][0]:itm['bits'][1] + 1]
            tv = itm['bits'][0]
        else:
            zo = b[itm['bits']]
            tv = itm['bits']
        new_row = pd.DataFrame([{'bits': itm['bits'], 'zo':zo, 'flag':k, 'Value':bits_dict[zo], 'tmp': tv}])
        df_decoded = pd.concat([df_decoded, new_row], axis=0, ignore_index=True)
    return df_decoded.sort_values('tmp').drop(['tmp'], axis = 1).reset_index(drop=True)


def lsc2_ds_scale(ds_fun, p, vr = True):
    # ! Quality band needs to be 'QA_PIXEL
    # for more details refer to:
    # table 5-1 of https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf
    # table 6-1 of https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat-8-9-C2-L2-ScienceProductGuide-v4.pdf
    
    data=[
          ['landsat_tm_c2_l2', f"{os.path.dirname(os.path.abspath(__file__))}/../aux/lsc2_scale_table_457.csv"],
          ['landsat_etm_c2_l2', f"{os.path.dirname(os.path.abspath(__file__))}/../aux/lsc2_scale_table_457.csv"],
          ['landsat_ot_c2_l2', f"{os.path.dirname(os.path.abspath(__file__))}/../aux/lsc2_scale_table_89.csv"],
         ]
    
    df = pd.DataFrame(data, columns=['product', 'table'])
    df = pd.read_csv(list(df.loc[df['product'] == p]['table'])[0])
    df['all_names'] = df['Band_Designation'] + ',' + df['Aliases']
    df['all_names'] = df.all_names.str.split(',')
    
    for v in ds_fun.var():
        if v != 'QA_PIXEL':
            v_row = df[df['all_names'].apply(lambda x: v in x)].iloc[0]

            # set nodata
            if not pd.isna(v_row.Fill_Value):
                ds_fun[v] = ds_fun[v].where(ds_fun[v] != v_row.Fill_Value)

            # restrict to valid range if requested
            if vr and not pd.isna(v_row.Valid_Range_min) and not pd.isna(v_row.Valid_Range_max):
                ds_fun[v] = ds_fun[v].where(np.all([ds_fun[v] >= v_row.Valid_Range_min,
                                                    ds_fun[v] <= v_row.Valid_Range_max], axis = 0))

            # Apply Multiplicative_Scale_Factor
            if not pd.isna(v_row.Multiplicative_Scale_Factor):
                ds_fun[v] = ds_fun[v] * v_row.Multiplicative_Scale_Factor
            # Apply Additive_Offset
            if not pd.isna(v_row.Additive_Offset):
                ds_fun[v] = ds_fun[v] + v_row.Additive_Offset
    
    return ds_fun


# def dms_to_dd(dms):
#     divs = [1, 60, 3600]
#     dd = 0
#     for i, v in enumerate(re.findall(r'\w+', dms)):
#         if v.isnumeric():    
#             dd = dd + int(v) / divs[i]
#         if v in ['S', 'W']:
#             dd = -dd
#     return dd 

def find_ipynb(search_string, search_dir = '../..', search_pattern = '', recent_first = None):
    """
    Description:
      Search (and link to) all .ipynb files containing a given <search_string>, optionally a <search_pattern> can be applied.
      e.g. find_ipynb(search_string = 'ndvi =')
             will list all scripts containing the string 'ndvi =' in the default directory (../..)
           find_ipynb(search_string = 'ndvi =', search_dir = '../../dea-notebooks', search_pattern = 'nalys')
             will list all scripts containing 'nalys' in their name and containing the string 'ndvi ='
             in ../../dea-notebooks
    -----
    Input:
      search_string: string to search for example
      search_dir (OPTIONAL): search path (current folder by default)
      search_pattern (OPTIONAL): string to filter result
      recent_first (OPTIONAL): sort by most recent if True, by oldest firt if False
    Output:
      List of scripts with a direct link, and the first line containing the <search_string>
    """
    
    css = """
          .highlight {background-color: yellow;}
          """
    display(HTML("<style>{}</style>".format(css)))

    
    fnames = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if (file.endswith('.ipynb')) and \
            not(file.endswith('-checkpoint.ipynb')) and \
            (search_pattern in file):
                fname = os.path.join(root,file)
                modtime = os.path.getmtime(fname)
                modtime_dt = datetime.datetime.fromtimestamp(modtime)
                fnames.append((fname, modtime_dt))
    
    if recent_first is not None:
        fnames.sort(key=lambda x: x[1], reverse = recent_first)
            
    for fname, modtime in fnames:
        with open(fname) as f:
            for line in f:
                if search_string in line and len(line) <= 160:
                    display(HTML('<a href="%s" target="_blank">%s</a> %s <br /> %s' % (
                        fname,
                        fname,
                        modtime.strftime("%Y-%m-%d %H:%M:%S"),
                        line.replace("\"","").replace(search_string, f'<span class="highlight">{search_string}</span>').strip()
                    )))

                    # display(HTML('<a href="%s" target="_blank">%s</a> %s <br /> %s' % (fname, fname, modtime.strftime("%Y-%m-%d %H:%M:%S"),line.replace("\"","").strip())))
                    break


def humanize_measure(m):
    humanized_name = ['red', 'green', 'blue', 'nir', 'swir_1', 'swir_2',
                      'aerosol_optical_thickness', 'coastal_aerosol', 'red_edge_1', 'red_edge_2', 'red_edge_3', 'water_vapour', 'narrow_nir', 'slc', 'water_vapour',
                      'surface_temperature', 'surface_temperature_quality']
    common_item = set(m.split(': ')[1].split(', ')).intersection(set(humanized_name))
    if len(common_item) == 1:
        return list(common_item)[0]
    else:
        return m.split(': ')[0]