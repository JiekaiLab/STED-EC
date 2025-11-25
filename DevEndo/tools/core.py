from pandas.io.formats.format import DataFrameFormatter
from sklearn.preprocessing import scale
from fbpca import pca
from scipy.linalg.blas import sgemm
from scipy.sparse import issparse
from collections import OrderedDict
from scipy.sparse import find, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import norm
from contextlib import contextmanager
from itertools import chain
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import concurrent.futures
import multiprocessing
import networkx as nx
import scipy.io as sio
import numpy as np
import scanpy as sc
import pandas as pd
import warnings
import anndata
import os
import re
mpl.rcParams['pdf.fonttype'] = 42


def checkFeatures(adata, features, blacklist = None):
    features = np.array(features)
    features = features[pd.Series(features).isin(adata.var.index.values)]
    if isinstance(blacklist, list) or isinstance(blacklist, tuple) or isinstance(blacklist, np.ndarray):
        features = features[~pd.Series(features).isin(blacklist)]
    
    fil_adata = adata[:,features]
    
    # variation = np.var(fil_adata.X.A, axis = 0)
    if issparse(fil_adata.X):
        variation = abs(fil_adata.X).sum(0).A[0] - abs(fil_adata.X.getrow(0).toarray()[0])
    else:
        variation = abs(fil_adata.X).sum(0) - abs(fil_adata.X[0,:])
    features = features[variation != 0]
    
    return(features)

def grep(pattern=None,X=None):
    p='.*'+pattern+'.*'
    if isinstance(X, str):
        matchObj=re.match(p,X)
        oX=matchObj.group()
    else:
        oX=[]
        for x in X:
            matchObj=re.match(p,x)
            if matchObj == None:
                continue
            oX.append(matchObj.group())
    return oX

def findBlackList(genes):
    """    
    This function find a list of gene names to detect common contaminant/housekeeping gene categories
    typically excluded from single-cell RNA-seq analysis: ribosomal (Rp/Rpl), mitochondrial (Mt-), 
    and hemoglobin (Hb) genes. The matching is case-insensitive for robust detection across naming conventions.

    Parameters
    ----------
    genes : list, array-like, or pandas.Series
        Input list/array of gene names to screen for blacklist genes.

    Returns
    -------
    numpy.ndarray
        1D array of blacklisted gene names (unique matches for ribosomal, mitochondrial, or hemoglobin genes).
        Preserves the order of detection (ribosomal → mitochondrial → hemoglobin).

    Examples
    --------
    >>> # Basic usage with gene list
    >>> gene_list = ['Rpl10', 'MT-CO1', 'HBA1', 'TP53', 'MRps27', 'ACTB']
    >>> blacklist = findBlackList(gene_list)
    >>> print(blacklist)
    ['Rpl10' 'MRps27' 'MT-CO1' 'HBA1']
    """
    genes = pd.Series(genes)
    ribo = grep('^([M]|)[Rr][Pp][SslL]', genes)
    mito = grep('^[Mm][Tt]-', genes)
    hb = grep('^H[Bb][AaBb]',genes)
    return(np.concatenate([ribo, mito, hb]))


def findHvgs(adata, 
             batches=None,
             min_disp=0.7,
             n_top_genes=None,
             blacklist='auto'):
    '''
    Parameters
    ----------
    adata : AnnData
        Input data.
    batches : str, optional
        The key in adata.obs_names used to find highly variable genes (hvg).
        Unique hvg of all batches will be returned.
    min_disp : float, optional
        cutoffs for the normalized dispersions.
    n_top_genes : None or int, optional
        Number of highly-variable genes to keep. 
    blacklist : None or str or tuple of strs, optional
        Unwanted genes, which will be excluded from hvgs.
        If this is 'auto', the mitochondrial and ribosomal genes 
        will be removed.
    
    Returns
    -------
    out : list
        Unique hvg for all batches.
    '''

    if isinstance(blacklist,str):
        if blacklist == 'auto':
            # Remove ribosomal and mitochondrial genes
            blacklist = findBlackList(adata.var_names)
    
    if batches is None:
        sc.pp.highly_variable_genes(adata, 
                                    min_mean=0.0125, 
                                    max_mean=3,
                                    min_disp=min_disp, 
                                    n_top_genes=n_top_genes)
        hvgs = adata.var[adata.var['highly_variable']].index.values

    else:
        unique_id = adata.obs[batches].unique()
        hvgs = np.array([])
        for x in unique_id:
            sub_adata = adata[adata.obs[batches] == x]
            sc.pp.highly_variable_genes(sub_adata, 
                                        min_mean=0.0125, 
                                        max_mean=3, 
                                        min_disp=min_disp, 
                                        n_top_genes=n_top_genes)
            sub_hvgs = sub_adata.var[sub_adata.var['highly_variable']].index.values
        hvgs = np.unique(np.concatenate([hvgs, sub_hvgs]))
    hvgs = checkFeatures(adata, hvgs, blacklist=blacklist)
    return(hvgs)


def choiceGroupCells(adata, groupby, cells_per_group=500, seed = 123):
    '''
    Random select n_samples cells from each group
    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The key of the observations grouping to consider.
    n_samples : Number of cells to choice.
    seed : Random seed.
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> deg = de.choiceGroupCells(adata, groupby='leiden', n_samples = 500)
    ------
    '''
    cells = adata.obs.index.values
    index = np.array([])
    group_list = adata.obs[groupby].unique()
    for x in group_list:
        pool = cells[adata.obs[groupby] == x]
        n_samples = min(len(pool), cells_per_group)
        np.random.seed(seed)
        index = np.concatenate([index, np.random.choice(pool, n_samples, replace=False)])
    return(adata[index])

def clean_cellid(adata):
    adata.obs_names = [re.sub(r'(-\d+)(-\d+)?$', '', s) for s in adata.obs_names]
    
def tfs(species = 'Mouse'):
    if species == 'Mouse':
        return(pd.read_csv('/data1/home/jiazi/lhlin/DATA/database/TFs/Mouse_TFs.csv')['Symbol'].values)
    if species == 'Human':
        return(pd.read_csv('/data1/home/jiazi/lhlin/DATA/database/TFs/Human_TFs.csv')['Symbol'].values)

def runPCA(adata,
           t1=None,
           t2=None, 
           n_components=20, 
           features=None, 
           n_top_genes=None, 
           rescale=True, 
           l2_norm=False,
           blacklist='auto',
           verbose=False):
    """
    Perform PCA on single-cell RNA-seq data with preprocessing (HVG selection, removing blacklist genes, scaling).
    
    This function streamlines PCA analysis for single-cell datasets by:
    1. Subsetting data to one/two time points/conditions (optional)
    2. Filtering blacklisted genes (ribosomal/mitochondrial/hemoglobin, auto-detected by default)
    3. Identifying highly variable genes (HVGs) (or using user-provided features)
    4. Rescaling data (optional)
    5. Running PCA and optionally applying L2 normalization to PCA embeddings
    6. Storing PCA results in the AnnData object's `.obsm['X_pca']`
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell gene expression data (raw or log-normalized).
        If using raw counts, ensure `.raw` is populated (used for HVG detection if `.X` is dense).
    t1 : bool array, optional
        First subset of cells to include (e.g., time point, condition). Can be:
        - Boolean array (mask) with length = number of cells
        - List of cell barcodes
        - String matching a categorical value in `adata.obs` (e.g., 'Day0')
        If None, uses all cells in `adata`.
    t2 : bool array, optional
        Second subset of cells to include (e.g., second time point). If provided, cells from `t1` and `t2`
        are concatenated before PCA. If None, only `t1` (or all cells) are used.
    n_components : int, default=20
        Number of principal components to compute.
    features : list, tuple, or np.ndarray, optional
        Predefined list of genes to use for PCA (bypasses HVG detection). If None, HVGs are identified
        for `t1` (and `t2` if provided) and their union is used.
    n_top_genes : int, optional
        Number of top highly variable genes to select (overrides `min_mean/max_mean/min_disp` in Scanpy's HVG function).
        If None, uses Scanpy's default threshold-based selection.
    rescale : bool, default=True
        Whether to scale the selected features to mean=0 and std=1 (recommended for PCA).
        Uses `scanpy.pp.scale` with `max_value=10` to cap extreme values.
    l2_norm : bool, default=False
        Whether to apply L2 normalization (unit norm) to the PCA embeddings (each cell's PCA vector has norm=1).
        Useful for downstream analyses like k-NN or clustering.
    blacklist : 'auto' or list/array, default='auto'
        Genes to exclude from analysis:
        - 'auto': Automatically detect ribosomal/mitochondrial/hemoglobin genes via `findBlackList`
        - List/array: Custom list of gene names to exclude
    verbose : bool, default=False
        Whether to print progress messages (HVG detection, scaling, PCA steps).

    Returns
    -------
    anndata.AnnData
        Updated AnnData object with:
        - `.obsm['X_pca']`: PCA embeddings (shape = [n_cells, n_components])
        - `.var['PCA_features']`: Boolean column indicating which genes were used for PCA
        If `t2` is provided, returns a concatenated AnnData object of `t1` + `t2` cells.

    Examples
    --------
    >>> # Basic PCA with auto HVG selection and blacklisting
    >>> adata = sc.read_h5ad('single_cell_data.h5ad')
    >>> adata_pca = runPCA(adata, t1='Day0', n_components=10, verbose=True)

    >>> # PCA with custom features and L2 normalization
    >>> custom_genes = ['TP53', 'MYC', 'EGFR']
    >>> adata_pca = runPCA(
    ...     adata,
    ...     t1='Day0',
    ...     t2='Day7',
    ...     features=custom_genes,
    ...     l2_norm=True,
    ...     blacklist=['MT-CO1', 'Rpl10']  # Custom blacklist
    ... )

    >>> # PCA with top 2000 HVGs and no rescaling
    >>> adata_pca = runPCA(
    ...     adata,
    ...     n_top_genes=2000,
    ...     rescale=False,
    ...     n_components=20
    ... )
    """
    if t1 is None:
        adata1 = adata
    else:
        adata1 = adata[t1] 
    if t2 is not None:
        adata2 = adata[t2]
        
    if blacklist == 'auto':
        # Remove ribosomal and mitochondrial genes
        blacklist = findBlackList(adata.var_names)
        
    # --- Find HVGs for each day
    if isinstance(features, list) or isinstance(features, tuple) or isinstance(features, np.ndarray):
        pass
    else:
        if verbose:
            print('Find hvg ...')
        if issparse(adata1.X):
            sc.pp.highly_variable_genes(adata1, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features1 = adata1.var[adata1.var['highly_variable']].index.values
        else:
            temp1 = anndata.AnnData(adata1.raw.X, obs = adata1.obs, var = adata1.raw.var)
            sc.pp.highly_variable_genes(temp1, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features1 = temp1.var[temp1.var['highly_variable']].index.values
            
        if t2 is None:
            features = checkFeatures(adata1, features1, blacklist)
        else:
            if issparse(adata2.X):
                sc.pp.highly_variable_genes(adata2, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
                features2 = adata2.var[adata2.var['highly_variable']].index.values
            else:
                temp2 = anndata.AnnData(adata2.raw.X, obs = adata2.obs, var = adata2.raw.var)
                sc.pp.highly_variable_genes(temp2, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
                features2 = temp2.var[temp2.var['highly_variable']].index.values
                
            features = np.union1d(features1, features2)
            features = checkFeatures(adata1, features, blacklist)
            features = checkFeatures(adata2, features, blacklist)
                
    if t2 is not None:
        adata1 = adata1.concatenate(adata2)
    sub_adata = adata1[:,features]

    if rescale:
        if verbose:
            print('Scale data of {} hvg ...'.format(len(features)))
        sc.pp.scale(sub_adata, max_value=10)
        
    # PCA
    if verbose:
        print('Run pca ...')
    adata1.obsm['X_pca'] = _pca(sub_adata.X, n_components)
    if l2_norm:
        # L2-norm for each cell
        adata1.obsm['X_pca'] = normalize(adata1.obsm['X_pca']) 
    adata1.var['PCA_features'] = adata1.var.index.isin(features)
    return(adata1)

# 
def runCCA(adata, 
           t1, 
           t2,
           n_components=20,
           features=None,
           n_top_genes=None,
           use_rep = None,
           use_pcs=None, 
           standarize=True,
           rescale=True,
           l2_norm=False,
           blacklist='auto',
           verbose=False):
    """
    Perform Seurat Canonical Correlation Analysis (CCA) as python implementation to integrate two single-cell datasets. Refer to "Integrating single-cell transcriptomic data across different conditions, technologies, and species"  Nature Biotechnology, 2018
    
    This function streamlines CCA for integrating two cell populations (e.g., time points, conditions, batches)
    from a single AnnData object, with preprocessing steps optimized for single-cell RNA-seq data:
    1. Subsets data to two target cell groups (t1 and t2)
    2. Filters blacklisted genes (ribosomal/mitochondrial/hemoglobin, auto-detected by default)
    3. Identifies highly variable genes (HVGs) (or uses user-provided features)
    4. Rescales/standardizes data (optional)
    5. Runs CCA (either on raw expression, PCA embeddings, or custom representations)
    6. Stores CCA embeddings in the concatenated AnnData object's `.obsm['X_cca']`

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell gene expression data (raw or log-normalized).
        If using raw counts, ensure `.raw` is populated (used for HVG detection/scaling if `.X` is dense).
    t1 : str, list, or bool array
        First cell group to integrate (e.g., time point 'Day0', condition 'Control'). Can be:
        - Boolean array (mask) with length = number of cells
        - List of cell barcodes
        - String matching a categorical value in `adata.obs`
    t2 : str, list, or bool array
        Second cell group to integrate (e.g., time point 'Day7', condition 'Treated'). 
        Must be in the same format as `t1`.
    n_components : int, default=20
        Number of canonical components to compute (dimensionality of CCA embeddings).
    features : list, tuple, or np.ndarray, optional
        Predefined list of genes to use for CCA (bypasses HVG detection). If None, HVGs are identified
        for `t1` and `t2` separately, and their union is used (filtered for blacklisted genes).
    n_top_genes : int, optional
        Number of top highly variable genes to select (overrides `min_mean/max_mean/min_disp` in Scanpy's HVG function).
        If None, uses Scanpy's default threshold-based HVG selection.
    use_rep : str, optional
        Key in `.obsm` of `adata` to use as input for CCA (e.g., 'X_pca', 'X_umap') instead of raw expression.
        If provided, CCA is run on this precomputed representation (instead of gene expression).
    use_pcs : int, optional
        Number of principal components to use if:
        - `use_rep=None`: PCA is run first on concatenated data, and top `use_pcs` PCs are used for CCA
        - `use_rep!=None`: Only the first `use_pcs` dimensions of the precomputed representation are used
        If None, uses all dimensions of the input representation (or skips PCA if `use_rep=None`).
    standarize : bool, default=True
        Whether to standardize (z-score) the input data (expression/PCA/representation) before CCA.
        Recommended for CCA to ensure features are on the same scale.
    rescale : bool, default=True
        Whether to scale gene expression to mean=0 and std=1 (via `scanpy.pp.scale`, max_value=10 to cap outliers)
        for the selected HVGs/features. Automatically set to True if `adata.X` is sparse (HVG detected from sparse data).
    l2_norm : bool, default=False
        Whether to apply L2 normalization (unit norm) to the final CCA embeddings (each cell's CCA vector has norm=1).
        Useful for downstream analyses like k-NN or clustering.
    blacklist : 'auto' or list/array, default='auto'
        Genes to exclude from analysis:
        - 'auto': Automatically detect ribosomal/mitochondrial/hemoglobin genes via `findBlackList`
        - List/array: Custom list of gene names to exclude
    verbose : bool, default=False
        Whether to print progress messages (HVG detection, scaling, standardization, CCA steps).

    Returns
    -------
    anndata.AnnData
        Concatenated AnnData object containing cells from `t1` and `t2` with:
        - `.obsm['X_cca']`: CCA embeddings (shape = [n_cells_t1 + n_cells_t2, n_components])
        - `.var['CCA_features']`: Boolean column indicating which genes were used for CCA preprocessing
        Preserves all original annotations from the input `adata` for the subset cells.


    Dependencies:
    - Requires `findBlackList` (blacklisted gene detection) and `checkFeatures` (feature validation)
    - Uses Scanpy for HVG detection/scaling and scikit-learn for PCA/CCA/normalization

    Examples
    --------
    >>> # Basic CCA integration of two time points (Day0 and Day7)
    >>> adata = sc.read_h5ad('single_cell_data.h5ad')
    >>> adata_cca = de.runCCA(
    ...     adata,
    ...     t1='Day0',
    ...     t2='Day7',
    ...     n_components=15,
    ...     n_top_genes=2000,
    ...     verbose=True
    ... )

    """
    adata1 = adata[t1]
    adata2 = adata[t2]

    merge_adata1 = adata1.concatenate(adata2)
    
    if blacklist == 'auto':
        # Remove ribosomal and mitochondrial genes
        blacklist = findBlackList(adata.var_names)

    if isinstance(features, list) or isinstance(features, tuple) or isinstance(features, np.ndarray):
        pass
    else:
        if verbose:
            print('Find hvg ...')
        if issparse(adata1.X):    
            sc.pp.highly_variable_genes(adata1, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features1 = adata1.var[adata1.var['highly_variable']].index.values
            rescale = True
        else:
            temp1 = anndata.AnnData(adata1.raw.X, obs = adata1.obs, var = adata1.raw.var)
            sc.pp.highly_variable_genes(temp1, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features1 = temp1.var[temp1.var['highly_variable']].index.values
            
        if issparse(adata2.X):
            sc.pp.highly_variable_genes(adata2, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features2 = adata2.var[adata2.var['highly_variable']].index.values
            rescale = True
        else:
            temp2 = anndata.AnnData(adata2.raw.X, obs = adata2.obs, var = adata2.raw.var)
            sc.pp.highly_variable_genes(temp2, min_mean=0.0125, max_mean=3, min_disp=0.7, n_top_genes=n_top_genes)
            features2 = temp2.var[temp2.var['highly_variable']].index.values
        features = np.union1d(features1, features2)
    
    features = checkFeatures(adata1, features, blacklist)
    features = checkFeatures(adata2, features, blacklist)

    
    adata1 = adata1[:,features]
    adata2 = adata2[:,features]
    merge_adata = merge_adata1[:,features]
    
    if rescale:
        if verbose:
            print('Scale data of {} hvg ...'.format(len(features)))
        if not issparse(adata1.X):
            merge_adata = anndata.AnnData(merge_adata.raw.X, obs = merge_adata.obs, var = merge_adata.raw.var)
            merge_adata = merge_adata[:,features]
            
            
        #sc.pp.scale(merge_adata, max_value=10)
        #adata1 = merge_adata[adata1.obs.index.values]
        #adata2 = merge_adata[adata2.obs.index.values]
            adata1 = anndata.AnnData(adata1.raw.X, obs = adata1.obs, var = adata1.raw.var)
            adata2 = anndata.AnnData(adata2.raw.X, obs = adata2.obs, var = adata2.raw.var)
            adata1 = adata1[:,features]
            adata2 = adata2[:,features]
        sc.pp.scale(adata1, max_value=10)
        sc.pp.scale(adata2, max_value=10)
    
    # Standarize
    if verbose:
        if standarize:
            print('Standarize ...')
    if use_rep is None:
        if use_pcs is None:
            if standarize:
                X1 = _scale(adata1.X)
                X2 = _scale(adata2.X)
            else:
                X1 = adata1.X
                X2 = adata2.X
        else:
            #if rescale:
            #    sc.pp.scale(merge_adata, max_value=10)
            X_pca = _pca(merge_adata.X, n_components=use_pcs)
            merge_adata.obsm['X_pca'] = X_pca
            if standarize:
                X1 = _scale(merge_adata[adata1.obs.index.values].obsm['X_pca'])
                X2 = _scale(merge_adata[adata2.obs.index.values].obsm['X_pca'])
            else:
                X1 = merge_adata[adata1.obs.index.values].obsm['X_pca']
                X2 = merge_adata[adata2.obs.index.values].obsm['X_pca']
    else:
        if use_pcs is None:
            use_pcs = merge_adata.obsm[use_rep].shape[1]
        else:
            use_pcs = min(use_pcs, adata1.obsm[use_rep].shape[1])
        if standarize:    
            X1 = _scale(adata1.obsm[use_rep][:,:use_pcs])
            X2 = _scale(adata2.obsm[use_rep][:,:use_pcs])
        else:
            X1 = adata1.obsm[use_rep][:,:use_pcs]
            X2 = adata2.obsm[use_rep][:,:use_pcs]
        
    # CCA
    if verbose:
        print('Run cca ...')
    merge_adata1.obsm['X_cca'] = _cca(X1, X2, n_components)
    if l2_norm:
        # L2-norm for each cell
        merge_adata1.obsm['X_cca'] = normalize(merge_adata1.obsm['X_cca']) 
    merge_adata1.var['CCA_features'] = merge_adata1.var.index.isin(features)
    
    return(merge_adata1)

def _scale(X):
    if issparse(X):
        return(scale(X.A,axis = 1, with_mean=True, with_std=True))
    else:
        return(scale(X,axis = 1, with_mean=True, with_std=True))

def _pca(X, n_components):
    np.random.seed(123)
    U, s, Vt = pca(X, k=n_components, raw=True)
    X_dimred = U[:, :n_components] * s[:n_components]
    return(X_dimred)

def _cca(X1, X2, n_components, L2_norm=False):
    X = sgemm(1, a=X1, b = X2.T)
    # X = np.matmul(X1, X2.T)
    np.random.seed(123)
    U, s, Vt = pca(X, k=n_components, raw=True) # increase iteration
    # cca_svd = np.vstack((U*s, Vt.T*s))
    cca_svd = np.vstack((U, Vt.T))
    # cca_svd = np.apply_along_axis(resign, 0, cca_svd)
    return(cca_svd)


def cat(df, cats, return_val = False):
    '''
    Parameters:
    ----------
    df : DataFrame
    cats : If cats is str, show the element of cats. If cats is dict, assign categories to df[cats] 

    Return:
    ----------
    Categorical dataframe
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> de.cat(deg, 'condition')    
    >>> deg = de.cat(deg, {'condition': ['A','B']})    
    '''
    if isinstance(cats, dict):
        for key in cats:
            df[key] = pd.Categorical(np.array(df[key]), cats[key])
        return(df)
    else:
        if isinstance(cats, str):
            if pd.api.types.is_categorical_dtype(df[cats]):
                val = df[cats].cat.categories.tolist()
                if return_val:
                    return(val)
                else:
                    print("'%s' is categories: %s"%(cats,val))
            else:
                val = df[cats].unique()
                if return_val:
                    return(val)
                else:
                    print("'%s' is not categories with %s"%(cats, val))



def factorOrder(df, key=None):
    '''
    Extract the unique elements in the columns of dataframe
    '''
    if key == None:
        data = df
    else:
        data = df[key]
    if pd.api.types.is_categorical_dtype(data):
        return(data.cat.categories.tolist())
    else:
        return(data.unique())

def log2count(adata,total_counts = None):
    """
    Reverse log-normalized expression values back to raw count values.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object containing log-normalized expression matrix (X) as a CSR sparse matrix.
        Observation (obs) and feature (var) metadata are retained in the output.
    total_counts : float, optional
        Target total counts per cell (library size) for scaling. If None, I will infer the size factor automatically. 

    Returns
    -------
    anndata.AnnData
        AnnData object with raw, integer count values in the X matrix (CSR sparse format),
        retaining the original obs and var metadata from the input.

    """
    matrix = adata.X.copy()
    matrix.data = np.expm1(matrix.data)
    norm_expr_total = np.array(matrix.sum(1)).flatten()
    size_factor_total = norm_expr_total / total_counts
    indptr = matrix.indptr
    data = matrix.data
    indices = matrix.indices
    
    for i in range(matrix.shape[0]):
        start = indptr[i]
        end = indptr[i+1]
        
        if start == end:
            continue
        
        row_data = data[start:end]
        scale_factor = np.min(row_data)
        if total_counts is not None:
            size_factor = size_factor_total[i]
        if size_factor == 0:
            continue
        data[start:end] = row_data / size_factor
    X = csr_matrix((data, indices, indptr), shape=matrix.shape)
    X.data = np.round(X.data).astype(int)
    return(anndata.AnnData(X = X, obs=adata.obs, var = adata.var))


def csr_row_mins(sparse_matrix):
    """Get the minimum value of non-zero elements for each row in a sparse matrix.
    
    Parameters
    ----------
    sparse_matrix : scipy.sparse.csr_matrix or sparse matrix-like
        Input sparse matrix (will be converted to CSR format if not already).
        
    Returns
    -------
    numpy.ndarray
        1D array of shape (n_rows,) where each element is:
        - The minimum non-zero value of the corresponding row if the row has non-zeros
        - NaN if the row contains no non-zero elements

    """
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()
        
    n_rows = sparse_matrix.shape[0]
    result = np.full(n_rows, np.nan)
    
    for i in range(n_rows):
        start_idx = sparse_matrix.indptr[i]
        end_idx = sparse_matrix.indptr[i+1]
        row_data = sparse_matrix.data[start_idx:end_idx]
        
        if len(row_data) > 0:
            result[i] = np.min(row_data)
            
    return result
