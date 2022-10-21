import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import warnings
import time
from sklearn.preprocessing import QuantileTransformer

## To do:
# include log
# include documentation
# remove unneccesary files for github
# Add to github
# Add summary staitics
# Read papers

def read_nii(nii_file, stats):
    ## NEED TO DOCUMENT HERE
    # Read in user defined file 
    print(f'Reading in {nii_file}')
    nii_img = nib.load(nii_file)

    # Check images have the same dimensions
    if len(nii_img.shape) != 3:
        return None

    if nii_img.shape != stats.shape:
        print(f'Resampling {nii_file} (shape: {nii_img.shape}) to match assocation stats (shape: {stats.shape})')
        nii_img = processing.resample_from_to(nii_img, stats)
    
    return nii_img

def read_and_mask(nifti_file, weights, mask):
    ## NEED TO DOCUMENT HERE
    try:
        nii_img = read_nii(nifti_file, weights)
        return nii_img.get_fdata()[mask]
    except:
        warnings.warn(f'Error parsing {nifti_file}, filling with NA')
        numb_el = mask.sum().sum().sum()
        return np.full((numb_el, ), np.nan)
    

def check_reg(nii_file, out):
    ### Function reads in nii file and plots out to file overlaying PVS to check alignment
    # nii_file: (str) path to nii_file to generate overlay on top of
    # out: (str) path to output for generated file, if None then will output to same directory as nii_file

    if nii_file.endswith('.nii') and  nii_file.endswith('.nii.gz'):
        raise ValueError('Please pass NIFTI filepath')

    # Read in association statistics
    tstat_file = os.path.join(os.path.dirname(__file__), 'tstats.nii')
    tstats = nib.load(tstat_file)
    tstats_data = tstats.get_fdata()
    ind = abs(tstats_data[:, :, :].T)<3.5
    tstats_data.T[ind] = np.nan

    # Extract image
    nii_img = read_nii(nii_file, tstats)
    if nii_img is None:
        raise ValueError(f'{nii_file} is not 3d image')

    nii_img_data = nii_img.get_fdata()

    # Generate plot
    print('Generating plot and overlay')
    alpha = 0.8
    cm_stats = 'jet'

    axial_slcs = [85, 80, 70, 60, 45, 40]
    n_cols = int(np.ceil(len(axial_slcs)/2))
    fig, axes = plt.subplots(2, n_cols, figsize=(16, 10))
    for ind, axial_slc in enumerate(axial_slcs):
        i, j = np.unravel_index(ind, (2, n_cols))
        axes[i, j].imshow(nii_img_data[:, :, axial_slc].T, cmap="gray", origin="lower")
        axes[i, j].imshow(tstats_data[:, :, axial_slc].T, cmap=cm_stats, origin="lower", alpha=alpha)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

    # If out is not defined then use nii_file path
    if out is None:
        out_parts = os.path.split(nii_file)
        out = os.path.join(out_parts[0], out_parts[1].split('.')[0] + '_reg_check.pdf')
        print(f'No user defined output, saving to {out}')
    
    plt.tight_layout()
    plt.savefig(out)

def compute_pvs(nii_filepaths, out):
    ## NEED TO DOCUMENT HERE

    # Read in text files
    nii_files = pd.read_csv(nii_filepaths, header=None).iloc[:, 0]
    n_files = len(nii_files)
    print(f'Read {n_files} from {nii_filepaths}')

    # Read in weights and generate mask
    print('Loading weights file')
    weight_file = os.path.join(os.path.dirname(__file__), 'weights.nii')
    weights = nib.load(weight_file)
    weights_data = weights.get_fdata()
    mask = (weights_data[:, :, :] != 0)
    wvec = weights_data[mask]
    
    # Read in nifti files (applying mask)
    print('Reading and masking images')
    masked_images = nii_files.apply(lambda x: read_and_mask(x, weights, mask))
    masked_images = np.stack(masked_images.values)
    
    # Generate PolyVoxel Score
    print('Computing Polyvoxel Score')
    pvs = np.matmul(masked_images, wvec)
    
    # Identify Outliers
    if len(pvs)>50:
        pvs = pd.Series(pvs)
        q1 = pvs.quantile(0.25)
        q3 = pvs.quantile(0.75)
        IQR=q3-q1
        outliers = ((pvs<(q1-1.5*IQR)) | (pvs>(q3+1.5*IQR)))
        pvs = pvs.values
        if sum(outliers)>0:
            outlier_files = nii_files[outliers].tolist()
            outlier_file_str = '\n\t'.join(outlier_files)
            warnings.warn(f'{sum(outliers)} outliers detected:\n\t{outlier_file_str}\n suggest running --check_reg on these files')
    else:
        outliers = np.full((n_files, ), False)
    qt = QuantileTransformer(output_distribution='normal')
    pvs_QT = np.full((n_files, ), np.nan)
    pvs_QT[~outliers] = qt.fit_transform(pvs[~outliers, np.newaxis]).squeeze()
    pvs = pd.DataFrame({'Scan': nii_files.tolist(), 'PVS': pvs, 'PVS_QT': pvs_QT})

    # Save out results
    if out is None:
        out = './pvs.tsv'
    print(f'Saving PVS to {out}')
    pvs.to_csv(out, sep='\t', index=False, na_rep='NA')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line tool generates a PolyVoxel Score (PVS) of the "Hemochromatosis Brain" using T2-Weighted NIFTI scans')
    parser.add_argument('--check_reg', help='Pass .nii file to check if it is registered with weights and MNI space before computing PVS', type=str, default=None)
    parser.add_argument('--nii_files', help='Pass path to text file containing filepaths to .nii files registered to MNI space', type=str, default=None)
    parser.add_argument('--out', help='Path to output', type=str, default=None)

    opt = parser.parse_args()

    if not opt.check_reg is None:
        check_reg(opt.check_reg, opt.out)
    elif not opt.nii_files is None:
        compute_pvs(opt.nii_files, opt.out)
    