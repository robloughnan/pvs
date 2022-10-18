import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd

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

def read_and_mask(nifti_file, mask):
    ## NEED TO DOCUMENT HERE
    pass
    

def check_reg(nii_file, out):
    ### Function reads in nii file and plots out to file overlaying PVS to check alignment
    # nii_file: (str) path to nii_file to generate overlay on top of
    # out: (str) path to output for generated file, if None then will output to same directory as nii_file

    if nii_file.endswith('.nii') and  nii_file.endswith('.nii.gz'):
        raise ValueError('Please pass NIFTI filepath')

    # Read in association statistics
    tstats = nib.load('./t_stats.nii')
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

def compute_pvs(nii_files, out):
    ## NEED TO DOCUMENT HERE

    # Read in text files
    nii_files = pd.read_csv(nii_files, header=None).iloc[:, 0]

    # Read in tstats in case scans need reslicing
    tstats = nib.load('./t_stats.nii')
    # Generate mask of most significant voxels

    # Read in nifti files (applying mask)
    masked_images = nii_files.apply(lambda x: read_and_mask(x, mask))

    # Generate PolyVoxel Score
    pvs = masked_images * weights
    pvs = pd.DataFrame({'Scan': nii_files.tolist(), 'PVS': pvs})

    # Save out results
    if out is None:
        out = './pvs.tsv'
    pvs.to_csv(out, sep='\t')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line tool generates a PolyVoxel Score (PVS) of the "Hemochromatosis Brain" using T2-Weighted NIFTI scans')
    parser.add_argument('--check_reg', help='Pass .nii file to check if it is registered with weights and MNI space before computing PVS', type=str, default=None)
    parser.add_argument('--nii_files', help='Pass path to text file containing filepaths to .nii files registered to MNI space', type=str, default=None)
    parser.add_argument('--out', help='Path to output', type=str, default=None)


    opt = parser.parse_args()

    if not opt.check_reg is None:
        check_reg(opt.check_reg, opt.out)
    elif not opt.nii_files is None:
        raise NotImplementedError('Computation of PVS not yet developed')
        # compute_pvs(opt.nii_files, opt.out)
    