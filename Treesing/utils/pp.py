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
from matplotlib import colors
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

###############################################################################            


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

# TODO Find blacklist for human genes
def findBlackList(genes):
    genes = pd.Series(genes)
    ribo = grep('^([M]|)[Rr][Pp][SslL]', genes)
    mito = grep('^[Mm][Tt]-', genes)
    hb = grep('^H[Bb][AaBb]',genes)
    return(np.concatenate([ribo, mito, hb]))

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
        features = checkFeatures(adata1, features, blacklist)
        if t2 is not None:
            features = checkFeatures(adata2, features, blacklist)
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

    adata1 = adata[t1]
    adata2 = adata[t2]

    merge_adata = adata1.concatenate(adata2)
    merge_adata.obs_names = np.concatenate([adata1.obs_names.values, adata2.obs_names])
    
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
    merge_adata = merge_adata[:,features]
    
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
            use_pcs = min(min(merge_adata.shape),use_pcs)
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
    merge_adata.obsm['X_cca'] = _cca(X1, X2, n_components)
    if l2_norm:
        # L2-norm for each cell
        merge_adata.obsm['X_cca'] = normalize(merge_adata.obsm['X_cca']) 
    merge_adata.var['CCA_features'] = merge_adata.var.index.isin(features)
    
    return(merge_adata)

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

# TODO: Add L2 norm
def _cca(X1, X2, n_components, L2_norm=False):
    X = sgemm(1, a=X1, b = X2.T)
    # X = np.matmul(X1, X2.T)
    np.random.seed(123)
    U, s, Vt = pca(X, k=n_components, raw=True) # increase iteration
    # cca_svd = np.vstack((U*s, Vt.T*s))
    cca_svd = np.vstack((U, Vt.T))
    # cca_svd = np.apply_along_axis(resign, 0, cca_svd)
    return(cca_svd)




def get_split_index_helper(str1, str2):
    '''
    Return: 
    1. partial list of str1 which are not contains str2
    2. partial list of str2 which are not contains str1
    '''
    # 初始化辅助列表和主列表
    sub_list1 = []
    main_list1 = []

    sub_list2 = []
    main_list2 = []

    # 使用for循环迭代整个数字列表
    for num in np.arange(len(str1)):
        if str1[num] in str2:
            if sub_list1:
                main_list1.append(sub_list1)
                sub_list1 = []
            sub_list2.append(num)
        else:
            sub_list1.append(num)

            if sub_list2:
                main_list2.append(sub_list2)
                sub_list2 = []

    # 添加末尾的数字子序列
    if sub_list1:
        main_list1.append(sub_list1)
    if sub_list2:
        main_list2.append(sub_list2)
    return(main_list1, main_list2)







def expMat(M, t):
    if t == 1:
        return(M)
    else:
        MM = M.dot(M)
        i = 2
        while i < t:
            MM = M.dot(MM)
            i += 1
    return(MM)

def fastPearsonr(X1,X2=None):
    warnings.filterwarnings('ignore')
    if X2 is None:
        return(sgemm(1, a=_scale(X1), b = _scale(X1).T)/X1.shape[1])
    else:
        return(sgemm(1, a=_scale(X1), b = _scale(X2).T)/X1.shape[1])

def snnFilter(nn, min_shared_neighbors=2):
    nn2 = nn.copy()
    nn2.data = np.repeat(1, len(nn2.data))
    snn = nn2.dot(nn2.T)
    snn.data[snn.data < min_shared_neighbors] = 0
    snn.eliminate_zeros()
    snn.data = np.repeat(1, len(snn.data))
    return(nn.multiply(snn))
    
def resign(x):
    if np.sign(x[0]) == -1:
        x = x*-1
    return(x)
        
def _convert_to_affinity(kNN, scaling_factors, with_self_loops=False):
    """ Convert adjacency matrix to affinity matrix
    """
    N = kNN.shape[0]
    rows, cols, dists = find(kNN)
    dists = dists ** 2/(scaling_factors.values[rows] ** 2)

    # Self loops
    if with_self_loops:
        dists = np.append(dists, np.zeros(N))
        rows = np.append(rows, range(N))
        cols = np.append(cols, range(N))
    aff = csr_matrix((np.exp(-dists), (rows, cols)), shape=[N, N])
    return(aff)

def dist2aff(kNN, adaptive_k=2):
    nodes = np.arange(kNN.shape[0])
    # Adaptive k
    scaling_factors = np.zeros(len(nodes))
    
    for i in np.arange(len(scaling_factors)):
        scaling_factors[i] = np.sort(kNN.data[kNN.indptr[i]:kNN.indptr[i + 1]])[adaptive_k - 1]

    scaling_factors = pd.Series(scaling_factors, index=nodes)

    # Affinity matrix
    nn_aff = _convert_to_affinity(kNN, scaling_factors, True)
    
    return(nn_aff)
    

def seqNN(adata,
             stage = 'stage',
             dm_method = 'auto',
             n_components = 'auto',
             n_neighbors = 'auto',
             use_rep = None, 
             use_pcs=None, 
             standarize=True,
             cca_merge_ratio = 0.5,
             min_n_components = 20,
             min_n_neighbors =10,
             max_n_components = 150,
             max_n_neighbors = 30,
             min_between_edge_prop=0.1,
             features = None, 
             aff_method = 'adaptive_gauss',
             rescale = True,
             blacklist = None,
             within_time_aggr = 'multiply',
             verbose = False):

    adata = adata.copy()         
    timepoints = adata.obs[stage]
    time_series = timepoints.unique()
    tp_cells = pd.Series()
    tp_offset = pd.Series()
    offset = 0
    for i in time_series:
        tp_offset[i] = offset
        tp_cells[i] = list(timepoints.index[timepoints == i])
        offset += len(tp_cells[i])
    
    rd_dict = OrderedDict()
    within_aff = OrderedDict()
    between_aff = OrderedDict()
    for i in np.arange(len(time_series)-1):
        
        id1 = time_series[i]
        id2 = time_series[i+1]
        ids = '{}_{}'.format(id1,id2)
        
        t1 = timepoints == id1
        t2 = timepoints == id2
        
        # Check n_components
        if n_components == 'auto':
            n_components = int(np.floor((t1.sum()+t2.sum())**(1/3))) 
        if n_components < min_n_components:
            n_components = min_n_components
        elif n_components > max_n_components:
            n_components = max_n_components
        
        # Check n_neighbors:
        if n_neighbors == 'auto':
            n_neighbors = int(np.floor((t1.sum()+t2.sum())**(1/3)))
        if n_neighbors < min_n_neighbors:
            n_neighbors = min_n_neighbors
        elif n_neighbors > max_n_neighbors:
            n_neighbors = max_n_neighbors
        
        if dm_method == 'auto':
            print('{}-nn graph for {} and {} on {} {} components'.format(n_neighbors, id1, id2, n_components, 'pca'))
            rd = 'pca'
            # Run pca first
            mdata = runPCA(adata,t1,t2,  
                           n_components=n_components,
                           features=features, 
                           rescale=rescale, 
                           blacklist = blacklist,
                           verbose = verbose)
            sc.pp.neighbors(mdata, n_neighbors = n_neighbors)
            nn_pca = mdata.uns['neighbors']['connectivities']
            conn = edgeStat(mdata, stage=stage)['between_edge_prop'].values[0]
            if conn < min_between_edge_prop:
                print('->-> edges proportion between {} and {} is {} lower than {}, run cca ...'.format(id1, id2, conn, min_between_edge_prop))
                rd = 'cca'
                mdata = runCCA(mdata,timepoints[t1].index.values,timepoints[t2].index.values,  
                               n_components=n_components,
                               use_rep = use_rep, 
                               use_pcs=use_pcs, 
                               standarize=standarize,
                               features=mdata.var.index.values[mdata.var['PCA_features']], 
                               rescale=False, 
                               blacklist = blacklist,
                               verbose = verbose)
        if dm_method == 'pca':
            print('{}-nn graph for {} and {} on {} {} components'.format(n_neighbors, id1, id2, n_components, dm_method))
            rd = dm_method
            mdata = runPCA(adata,t1,t2,  
                           n_components=n_components,
                           features=features, 
                           rescale=rescale, 
                           blacklist = blacklist,
                           verbose = verbose)
                           
        if dm_method == 'cca':
            print('{}-nn graph for {} and {} on {} {} components'.format(n_neighbors, id1, id2, n_components, dm_method))  
            rd = dm_method
            mdata = runCCA(adata,t1,t2, 
                           use_rep = use_rep, 
                           use_pcs=use_pcs, 
                           standarize=standarize,
                           n_components=n_components,
                           features=features, 
                           rescale=rescale, 
                           blacklist = blacklist,
                           verbose = verbose)
        
        
        # Nearest neighbors
        if (aff_method == 'gauss') or (aff_method == 'umap'):
            sc.pp.neighbors(mdata, use_rep = 'X_{}'.format(rd),method = aff_method, n_neighbors = n_neighbors)
            kNN = mdata.uns['neighbors']['connectivities']
        if aff_method == 'adaptive_gauss':
            sc.pp.neighbors(mdata, use_rep = 'X_{}'.format(rd),n_neighbors=n_neighbors)
            kNN = dist2aff(mdata.uns['neighbors']['distances'], 
                           adaptive_k = int(np.floor(n_neighbors/3)))
        if (dm_method == 'auto') & (rd == 'cca') & (cca_merge_ratio>0):
            print('->-> merge pca graph and cca graph ...')
            kNN = cca_merge_ratio*kNN + (1-cca_merge_ratio)*nn_pca
            
        rd_dict[ids] = mdata.obsm['X_{}'.format(rd)]
        
        # Split within time point nn and between time point nn
        nn1,nn2,bnn = splitNN(kNN, 
                      timepoints[np.concatenate((timepoints[t1].index.values, 
                                                 timepoints[t2].index.values))], 
                                                 id1, id2)
        
        # Average within time connectivities
        #if id1 in within_aff:
        #    if within_time_aggr == 'mean':
        #        within_aff[id1] = (within_aff[id1] + nn1)/2
        #    if within_time_aggr == 'multiply':
        #        within_aff[id1] = within_aff[id1].multiply(nn1).sqrt()
        
        if id1 in within_aff:
            pass
        else:
            within_aff[id1] = nn1
        #    
        #if id2 in within_aff:
        #    if within_time_aggr == 'mean':
        #        within_aff[id2] = (within_aff[id2] + nn2)/2
        #    if within_time_aggr == 'multiply':
        #        within_aff[id2] = within_aff[id2].multiply(nn2).sqrt()
        if id2 in within_aff:
            pass
        else:
            within_aff[id2] = nn2
        
        between_aff[id1] = bnn
        
    return({'rd_dict':rd_dict, 'within_aff':within_aff, 'between_aff':between_aff,'timepoints':timepoints})
       
def mergeNN(nn_info, filter_below_sd=None, min_degree = -1):
    timepoints = nn_info['timepoints']
    time_series = timepoints.unique()
    # tp_cells = pd.Series()
    tp_offset = pd.Series(dtype = np.int64)
    offset = 0
    for i in time_series:
        tp_offset[i] = offset
        # tp_cells[i] = list(timepoints.index[timepoints == i])
        offset += np.sum(timepoints.values == i, dtype = np.int64)
              
    N = nn_info['timepoints'].shape[0]
    # Within time graph
    within_nn = csr_matrix(([0], ([0], [0])), [N, N])
    for i in np.arange(len(time_series)):
        t = time_series[i]
        x, y, z = find(nn_info['within_aff'][t])
        x = x + tp_offset[t]
        y = y + tp_offset[t]
        nn = csr_matrix((z, (x, y)), shape=[N, N])
        if filter_below_sd != None:
            nn = filterNN(nn, filter_below_sd)
        within_nn += nn
    
    # Between time graph
    between_nn = csr_matrix(([0], ([0], [0])), [N, N])
    for i in np.arange(len(time_series)-1):
        t1 = time_series[i]
        t2 = time_series[i+1]
        x, y, z = find(nn_info['between_aff'][t1])
        x = x + tp_offset[t1]
        y = y + tp_offset[t2]
        nn = csr_matrix((z, (x, y)), shape=[N, N])
        if filter_below_sd != None:
            nn = filterNN(nn, filter_below_sd)
        between_nn += nn
    
    # Global graph
    global_nn = within_nn + between_nn + between_nn.T
    inliers = np.where(np.array([len(x) for x in global_nn.tolil().rows]) > min_degree)[0]
    
    return((global_nn, inliers))

def filterNN(nn, sd=-1):
    weight = nn.data
    m, v = norm.fit(weight)
    retain_idx = np.where(weight >= (m+sd*v))
    
    # New graph
    N = nn.shape[0]
    x, y, z = find(nn)
    x = x[retain_idx]
    y = y[retain_idx]
    z = z[retain_idx]
    return(csr_matrix((z, (x, y)), shape=[N, N]))

def splitNN(kNN, timepoints, key1, key2):
    t1_idx = np.arange(kNN.shape[0])[timepoints == key1]
    t2_idx = np.arange(kNN.shape[0])[timepoints == key2]
    n1 = len(t1_idx)
    n2 = len(t2_idx)
    nn1 = kNN[t1_idx,:][:,t1_idx]
    nn2 = kNN[t2_idx,:][:,t2_idx]
    bnn = kNN[t1_idx,:][:,t2_idx]
    return(nn1, nn2, bnn)

def plotEdgeStat(df,
                 cmap = 'plasma_r',
                 point_size=100,
                 linewidth=1.5,
                 figsize=(6,3)):
    norm = mpl.colors.Normalize(vmin=0, vmax=df['between_edge_prop'].max())
    cmap = plt.get_cmap('plasma_r')
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [m.to_rgba(x) for x in df['between_edge_prop']]
    seg = [[(i,i),(0,df['between_edge_prop'][i]),'k--'] for i in range(df.shape[0])]
    fig = plt.figure(figsize=figsize)
    plt.plot(*list(chain(*seg)),linewidth=linewidth,zorder=0)
    plt.plot(df['adj'],
             df['between_edge_prop'],
             c='#000000',
             linewidth=linewidth,
             zorder=1)
    plt.scatter(df['adj'], 
                df['between_edge_prop'],
                c = colors, 
                s=point_size,
                zorder=2)
    plt.colorbar(m)
    plt.grid(b=None)
    plt.xticks(rotation = 90,ha='center') 
    plt.xlabel('Adjacent time')
    plt.ylabel('Edge proportion between time')
    plt.show()

def edgeStat(adata, 
             stage='stage',
             aff = None,
             plot=True,cmap = 'plasma_r',
             point_size=100,
             linewidth=1.5,
             figsize=(6,3)):
    timepoints = adata.obs[stage]
    time_series = timepoints.unique()
    if aff is None:
        aff = adata.obsp['connectivities']
    within_edges = []
    between_edges = []
    ids_list1 = []
    ids_list2 = []
    merge_ids = []
    for i in np.arange(len(time_series) - 1):
        id1 = time_series[i]
        id2 = time_series[i+1]
        t1 = (timepoints == id1).values
        t2 = (timepoints == id2).values
        total_edges = len(aff[(t1|t2),:][:,(t1|t2)].data)
        
        within_edges.append((len(aff[t1,:][:,t1].data) + len(aff[t2,:][:,t2].data))/total_edges)
        between_edges.append((len(aff[t1,:][:,t2].data)+len(aff[t2,:][:,t1].data))/total_edges)
        ids_list1.append(id1)
        ids_list2.append(id2)
        merge_ids.append('{}-{}'.format(id1, id2))
    df = pd.DataFrame({'adj':merge_ids,
                         'id1':ids_list1,
                         'id2':ids_list2,
                         'within_edge_prop':within_edges,
                         'between_edge_prop':between_edges})
    if plot:
        plotEdgeStat(df)
    return(df)

def _toSymmetric(M):
    W = M.copy()  # need to copy the distance matrix here; what follows is inplace
    indices = OrderedDict()
    for i in np.arange(W.shape[0]):
        indices[i] = W[i].nonzero()[1]
    
    W = W.tolil()
    for i in indices:
        for j in indices[i]:
            if i not in set(indices[j]):
                W[j, i] = W[i, j]
    W = W.tocsr()
    return(W)
 
def trimNN(adata, dist_std=2, min_neighbors=3, inplace = False):
    A = adata.uns['neighbors']['connectivities'].copy()
    D = adata.uns['neighbors']['distances'].copy()
     
    thr = np.median(D.data)+dist_std*np.std(D.data)
    for i in np.arange(len(D.indptr)-1):
        ind = D.indices[D.indptr[i]:D.indptr[i+1]]
        d = D.data[D.indptr[i]:D.indptr[i+1]]
        d_sort = d.copy()
        d_sort.sort()
        d[d>=max(thr,d_sort[min_neighbors])] = 0
    D.eliminate_zeros()
    
    # D.data = np.array([1]*len(D.data))
    mask = _toSymmetric(D)
    mask.data = np.array([1]*len(mask.data))
    A = A.multiply(mask)
    if inplace:
        adata.uns['neighbors']['connectivities'] = A
        adata.uns['neighbors']['distances'] = D
    else:    
        return((A,D))



## Reviews PAGA
_adata = None

# @timer # 该修饰会取代函数本身的return
def revise_paga(adata,
                group = 'celltype_annotation_0726',
                edges = None,
                use_rep = None,
                use_connectivities = False,
                rd_method = 'pca',
                n_sample = 500,
                n_top_genes = 500,
                n_neighbors = 10,
                n_components = 10,
                thr = 0.1,
                seed = 123,
                plot_umap=True):
    
    adata = adata.copy()
    if edges is None:
        aff = adata.uns['paga']['connectivities'].copy()
        aff = aff.multiply(aff>thr)
        aff.eliminate_zeros()
        G = nx.from_scipy_sparse_matrix(aff)
        mapping = dict(zip(np.arange(aff.shape[0]), adata.obs[group].cat.categories))
        G = nx.relabel_nodes(G, mapping)
        edges = list(G.edges())
        
    cells = adata.obs.index.values
    index = np.array([])
    for x in adata.obs[group].unique():
        pool = cells[adata.obs[group] == x]
        sample = min(len(pool), n_sample)
        index = np.concatenate([index, np.random.choice(pool, sample, replace=False)])
        
    index = np.sort(index)
    adata = adata[index]
    
    global _adata
    _adata = adata
    # Parallel
    p = [(es, group, n_top_genes, use_rep, use_connectivities, rd_method, n_components,
            n_neighbors, plot_umap) for es in edges]
            
    with concurrent.futures.ProcessPoolExecutor() as executor: 
        results = executor.map(do_analysis, p) # args有特殊含义，不能用在这; map 无法传递复杂对象

    df = pd.concat([i for i in results])
    return(df)

'''
def do_analysis(x):
    es = x[0]
    group = x[1]
    n_top_genes = x[2]
    use_rep = x[3]
    use_connectivities = x[4]
    rd_method = x[5]
    n_components = x[6]
    n_neighbors = x[7]
    plot_umap = x[8]
    
    global _adata # adata 只能通过声明全局变量来使用，不能作为普通参数传输到函数中
    sub_adata = _adata[_adata.obs[group].isin(es)]
    if use_connectivities:
        stat = edgeStat(sub_adata, group)
    else:
        if use_rep is None:
            sc.pp.highly_variable_genes(sub_adata, n_top_genes=n_top_genes)
            sub_adata = sub_adata[:,sub_adata.var['highly_variable']]
            if rd_method == 'pca':
                sub_adata.obsm['X_pca'] = _pca(sub_adata.X,n_components = n_components)
            if rd_method == 'cca':
                pool1 = sub_adata.obs.index.values[sub_adata.obs[group]==es[0]]
                pool2 = sub_adata.obs.index.values[sub_adata.obs[group]==es[1]]
                sub_adata.obsm['X_pca'] = _cca(sub_adata[pool1].X.A,
                                               sub_adata[pool2].X.A,
                                               n_components = n_components)
        else:
            sub_adata.obsm['X_pca'] = sub_adata.obsm[use_rep]
        sc.pp.neighbors(sub_adata, n_neighbors = n_neighbors)
        if plot_umap:
            sc.tl.umap(sub_adata)
            sc.settings.set_figure_params(dpi=100, frameon=True, fontsize=12)
            sc.pl.umap(sub_adata, color = group, 
                       save='_{}_{}.png'.format(re.sub('/','-',es[0]),re.sub('/','-',es[1])))
        stat = edgeStat(sub_adata, group)
        
    # Normalize score
    n1 = (sub_adata.obs[group]==es[0]).sum()
    n2 = (sub_adata.obs[group]==es[1]).sum()
    
    size_factor = 1 - abs(n1-n2)/max(n1,n2)
    
    stat['between_edge_prop_adj'] = stat['between_edge_prop']/size_factor
    return(stat)
'''

























