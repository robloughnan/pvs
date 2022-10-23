# Introduction
This lightweight tool can be used to generate a PolyVoxel Score across a set of MNI registered T2-Weighted scans capturing the iron deposition seen in the architypal "Hemochromatosis Brain". This pattern of brain iron desposition may relevant for Parkinson's Disease risk. Two papers discussing details can be found here:

[Loughnan, R., et al. Association of Genetic Variant Linked to Hemochromatosis With Brain Magnetic Resonance Imaging Measures of Iron and Movement Disorders. JAMA Neurology, 2015.](https://jamanetwork.com/journals/jamaneurology/fullarticle/2794928) 

[BIORXIV](https://jamanetwork.com/journals/jamaneurology/fullarticle/2794928) 

# Getting Started

In order to download `pvs` you should clone this repository via the following command:

```
git clone https://github.com/robloughnan/pvs.git
cd pvs
```

To install dependancies (`nibabel`, `pandas`, `sklearn`, `tqdm`, `matplotlib` and `numpy`) you can use `conda` package manager availible [here.](https://store.continuum.io/cshop/anaconda/)

Once conda installed you can create a new environement with pvs dependencies.

```
conda env create --file environment.yml
source activate pvs
```

# How to Run

`pvs` has two simple functionalities:

1.  Provide plots of input scans overlaid with PVS weights to verify scans are aligned correctly (`--check_reg` option)
2. Generate PolyVertex Scores using weights generated from [this analysis](ref) on set of images (`--nii_files` option)

## Check Registration

`pvs` assumes that the T2-Weighted scans you are trying to generate PolyVoxel Scores for are already registered to MNI space. To check this you can run the following:

```
python pvs --check_reg INPUT_SCAN.nii --out out_reg_check.jpg
```

Replacing `INPUT_SCAN.nii` with your input T2-Weighted scan. This should generate a file that looks like this:

![alt text](https://github.com/robloughnan/pvs/blob/main/mni152_reg_check.jpg?raw=true)

This should enable you to verify that the weights are aligned with your image (i.e. peaks in the basal ganglia and cerebellum)


## Generate PolyVoxel Score
Using the option `--nii_files` you can pass the filepath to a text file containing a list of NIFTI files which you would like to generate PolyVoxel Scores for. An example of the contents of this file (named `nifti_files.txt`):
```
/path/to/scan_1.nii
/path/to/scan_2.nii
/path/to/scan_3.nii
/path/to/scan_4.nii
...
```

You can then generate PVS's for each of these scans using:
```
python pvs --nii_files nifti_files.txt --out PVS_out.tsv
```
```
Scan    PVS     PVS_QT
/path/to/scan1.nii        -26330944.293608118     1.1448100220071562
/path/to/scan2.nii        -24528804.382988364     1.883403395079479
/path/to/scan3.nii        -25013190.256748438     1.6765107907543326
/path/to/scan4.nii        -30374732.59083167      NA
...
```
The `PVS` column is the raw calculated PolyVoxel Score, this is generated for every file which can successfully be read in. The `PVS_QT` column is the quantile transformed PVS, here transformed to a normal distribution. Outliers are given `NA` values in this column: this could indicate that these scans are poorly registered (you may want to check their alignment with `--check_reg`). Outilers are those that are 1.5 x IQR (interquartile range) outside of the 25th and 75th percentiles.
## Citation

If you use this software please cite:

[Loughnan, R., et al. Association of Genetic Variant Linked to Hemochromatosis With Brain Magnetic Resonance Imaging Measures of Iron and Movement Disorders. JAMA Neurology, 2015.](https://jamanetwork.com/journals/jamaneurology/fullarticle/2794928) 

and 

[BIORXIV](https://jamanetwork.com/journals/jamaneurology/fullarticle/2794928) 
