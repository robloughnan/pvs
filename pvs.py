import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import logging
from tqdm import tqdm

def create_logger(log):
    """
    Creates a logger that logs to both a file and the console 
    Parameters
    ----------
        log: str 
            a file path to log to 
    Returns
    -------
        Logger
            the logger object
    """
    # Create logger
    logger = logging.getLogger('pvs')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log, mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return(logger)

def read_nii(nii_file, stats, verbose):
    """
    Reads in NIFTI file and resclices according to stats
    Parameters
    ----------
        nii_file: str 
            file path to nifti file to be read in
        stats: Nifti1Image
            nifti image defining how voxels should be resliced
        verbose: bool
            flag to determine if each file being read in is logged
    Returns
    -------
        img: Nifti1Image
            the image that is read in (and rescliced if necessary)
    """
    if verbose:
        logger.info(f'Reading in {nii_file}')
    nii_img = nib.load(nii_file)

    # Check images have the same dimensions
    if len(nii_img.shape) != 3:
        return None

    if nii_img.shape != stats.shape:
        if verbose:
            logger.info(f'Resampling {nii_file} (shape: {nii_img.shape}) to match assocation stats (shape: {stats.shape})')
        nii_img = processing.resample_from_to(nii_img, stats)
    
    return nii_img

def read_and_mask(nifti_file, weights, mask, verbose):
    """
    Reads in NIFTI file and resclices according to weights and applys mask 
    Parameters
    ----------
        nii_file: str 
            file path to nifti file to be read in
        weights: Nifti1Image
            nifti image defining how voxels should be resliced
        mask: np.array
            a boolean array indicating which voxels to be masked
        verbose: bool
            flag to determine if each file being read in is logged
    Returns
    -------
        np.array
            one dimensional array of read in image masked
    """
    try:
        nii_img = read_nii(nifti_file, weights, verbose)
        return nii_img.get_fdata()[mask]
    except:
        logger.warning(f'Error parsing {nifti_file}, filling with NA')
        numb_el = mask.sum().sum().sum()
        return np.full((numb_el, ), np.nan)
    

def check_reg(nii_file, modality, out):
    """
    Function reads in nii_file and plots out to file overlaying PVS (unregularized) weights to check alignment
    Parameters
    ----------
        nii_file: str 
            file path to nifti file to check regisration of
        modality: str
            modality to use for weights, can be t2 or t2star
        out: str
            path to output for generated file, if None then will output to same directory as nii_file
    Returns
    -------
        None
    """

    if nii_file.endswith('.nii') and  nii_file.endswith('.nii.gz'):
        raise ValueError('Please pass NIFTI filepath')

    # Read in association statistics
    tstat_file = os.path.join(os.path.dirname(__file__), 'tstats_' + modality + '.nii')
    tstats = nib.load(tstat_file)
    tstats_data = tstats.get_fdata()
    ind = abs(tstats_data[:, :, :].T)<3.5
    tstats_data.T[ind] = np.nan

    # Extract image
    nii_img = read_nii(nii_file, tstats, verbose=True)
    if nii_img is None:
        raise ValueError(f'{nii_file} is not 3d image')

    nii_img_data = nii_img.get_fdata()

    # Generate plot
    logger.info('Generating plot and overlay')
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
    
    logger.info(f'Saving to out')
    plt.tight_layout()
    plt.savefig(out)

def compute_pvs(nii_filepaths, modality, out, verbose):
    """
    Function reads in files from nii_filepaths and computes PolyVoxel Score
    Parameters
    ----------
        nii_filepaths: str 
            path to text file containing list of nifti files to compute polyvoxel score for
        modality: str
            modality to use for weights, can be t2 or t2star
        out: str
            path to output PVS to
        verbose: bool  
            if True will print out each file being read in
    Returns
    -------
        None
    """

    # Read in text files
    nii_files = pd.read_csv(nii_filepaths, header=None).iloc[:, 0]
    n_files = len(nii_files)
    logger.info(f'Read {n_files} from {nii_filepaths}')

    # Read in weights and generate mask
    logger.info('Loading weights file')
    weight_file = os.path.join(os.path.dirname(__file__), 'weights_' + modality + '.nii')
    weights = nib.load(weight_file)
    weights_data = weights.get_fdata()
    mask = (weights_data[:, :, :] != 0)
    wvec = weights_data[mask]
    
    # Read in nifti files (applying mask)
    logger.info('Reading and masking images')
    tqdm.pandas() # enable progress bar
    masked_images = nii_files.progress_apply(lambda x: read_and_mask(x, weights, mask, verbose))
    masked_images = np.stack(masked_images.values)
    
    # Generate PolyVoxel Score
    logger.info('Computing Polyvoxel Score')
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
            logger.warning(f'{sum(outliers)} outliers detected:\n\t{outlier_file_str}\n suggest running --check_reg on these files')
    else:
        outliers = np.full((n_files, ), False)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min([sum(~outliers), 1000]))
    pvs_QT = np.full((n_files, ), np.nan)
    pvs_QT[~outliers] = qt.fit_transform(pvs[~outliers, np.newaxis]).squeeze()
    pvs = pd.DataFrame({'Scan': nii_files.tolist(), 'PVS': pvs, 'PVS_QT': pvs_QT})

    logger.info(f'Saving PVS to {out}')
    pvs.to_csv(out, sep='\t', index=False, na_rep='NA')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line tool generates a PolyVoxel Score (PVS) of the "Hemochromatosis Brain" using T2-Weighted NIFTI scans')
    parser.add_argument('--check_reg', help='Pass .nii file to check if it is registered with weights and MNI space before computing PVS', type=str, default=None)
    parser.add_argument('--nii_files', help='Pass path to text file containing filepaths to .nii files registered to MNI space', type=str, default=None)
    parser.add_argument('--modality', help='Modality to use weights from can be t2 or t2star [t2]', type=str, default='t2')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--out', help='Path to output', type=str, default=None)

    opt = parser.parse_args()
    
    if not opt.modality in ['t2', 't2star']:
        raise ValueError('modality must be either t2 or t2star')

    if not opt.check_reg is None and not opt.nii_files is None:
        raise ValueError('Cannot select both --check_reg and --nii_files')
    
    if opt.check_reg is None and opt.nii_files is None:
        raise ValueError('Must select either --check_reg or --nii_files')
    
    # If out is not defined then use nii_file path
    if opt.out is None and not opt.check_reg is None:
        out_parts = os.path.split(opt.check_reg)
        log = os.path.join(out_parts[0], out_parts[1].split('.')[0] + 'pvs.reg_check.log')
        out = log.replace('.log', '.jpg')
        print(f'No user defined output, logging to to {log}')
    if opt.out is None and not opt.nii_files is None:
        log = './pvs.log'
        print(f'No user defined output, logging to to {log}')
        out = log.replace('.log', '.tsv')
    else:
        out = opt.out
        log = os.path.splitext(opt.out)[0] + '.log'
    
    global logger 
    logger = create_logger(log)
    
    # Log Arguments
    logger.info(f'{__file__} Starting')
    for arg, value in sorted(vars(opt).items()):
        logger.info("Argument --%s: %r", arg, value)
        
        
    if not opt.check_reg is None:
        check_reg(opt.check_reg, opt.modality, out)
    elif not opt.nii_files is None:
        compute_pvs(opt.nii_files, opt.modality, out, opt.verbose)
    logger.info('Finished')