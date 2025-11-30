# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:15:27 2025

@author: charlie
"""
import os,sys,re
import anndata, os
import numpy as np
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
import networkx as nx
import itertools
from tqdm import tqdm
from scipy.linalg.blas import sgemm
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.sparse import issparse
from scipy import interpolate
from scipy.stats import norm
from scipy.sparse import csr_matrix,csc_matrix, find
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist
from scipy import interpolate
from scipy.stats import entropy
import scanpy.external as sce
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
from contextlib import contextmanager
from pygam import GAM, s
import concurrent.futures
from scipy.spatial.distance import pdist
from scipy.special import comb
from itertools import combinations
from functools import reduce
from pandas.api.types import is_categorical_dtype
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hypergeom
## reload external modules
# import importlib
# import tools.comparison
# import tools.core
# importlib.reload(tools.core)
# importlib.reload(tools.comparison)
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
from tools.core import *
from tools.comparison import *
from tools.plot import *
# from tools.communication import *
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

def fast_corrcoef(X1, X2=None, decimals=3):
    '''
    Fast computation of Pearson correlation coefficient matrix (accelerated by standardization + BLAS-optimized matrix multiplication)

    Parameters
    ----------
    X1 : np.ndarray
        Input matrix 1 with dimension (n_samples_1, n_features):
    X2 : np.ndarray, optional
        Input matrix 1 with dimension (n_samples_2, n_features):
    decimals : int, optional
        Number of decimal places to retain in results, default is 3.

    Returns
    -------
    Pearson correlation coefficient matrix with dimension (n_samples_1, n_samples_1) or (n_samples_1, n_samples_2)
    '''
    if X2 is None:
        X2 = X1
    scaler = StandardScaler(with_mean=True, with_std=True)
    X1 = scaler.fit_transform(X1.T).T 
    X2 = scaler.fit_transform(X2.T).T
    return(np.round(sgemm(1, X1, X2.T)/X1.shape[1], decimals))

def fastKnn(X1, 
            X2=None, 
            n_neighbors=20, 
            metric='euclidean', 
            M=40, 
            post=0, # Buffer memory error occur when post != 0
            efConstruction=100,
            efSearch=200):
    '''
    Fast computation of knn using nmslib.

    Parameters
    ----------
    X1 : np.ndarray
        Input matrix 1 with dimension (n_samples_1, n_features):
    X2 : np.ndarray, optional
        Input matrix 1 with dimension (n_samples_2, n_features):
    n_neighbors : int
        Number of nearest neighbors

    Returns
    -------
    distances and knn indices between samples.
    '''
    if metric == 'euclidean':
        metric = 'l2'
    if metric == 'cosine':
        metric = 'cosinesimil'
    if metric == 'jaccard':
        metric = 'bit_jaccard'
    if metric == 'hamming':
        metric = 'bit_hamming'
    # efConstruction: improves the quality of a constructed graph but longer indexing time
    index_time_params = {'M': M,
                         'efConstruction': efConstruction, 
                         'post' : post} 
    # efSearch: improves recall at the expense of longer retrieval time
    efSearch = max(n_neighbors, efSearch)
    query_time_params = {'efSearch':efSearch}
    
    if issparse(X1):
        if '_sparse' not in metric:
            metric = f'{metric}_sparse'
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.SPARSE_VECTOR)
    else:
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(X1)
    index.createIndex(index_time_params, print_progress=False)
    index.setQueryTimeParams(query_time_params)
    if X2 is None:
        neighbours = index.knnQueryBatch(X1, k=n_neighbors)
    else:
        neighbours = index.knnQueryBatch(X2, k=n_neighbors)
    
    distances = []
    indices = []
    for i in neighbours:
        if len(i[0]) != n_neighbors:
            vec_inds = np.zeros(n_neighbors)
            vec_dist = np.zeros(n_neighbors)
            vec_inds[:len(i[0])] = i[0]
            vec_dist[:len(i[1])] = i[1]
            indices.append(vec_inds)
            distances.append(vec_dist)        
        else:
            indices.append(i[0])
            distances.append(i[1])
    distances = np.vstack(distances)
    indices = np.vstack(indices)
    indices = indices.astype(int)
    if metric == 'l2':
        distances = np.sqrt(distances)
    
    return(distances, indices)

def score_genes(adata, 
               geneset = None,
               method = 'mean', # scanpy
               normalize = False,
               uns_keys = 'score',
               **kwargs):
    '''
    Scoring single cells for pathway. Equal to mean of expression of a set of genes.
    
    Parameters:
    ----------
    adata: anndata.AnnData.
    geneset: Dict of gene set.
    method: mean of gene expression, or scanpy method
    uns_key: The place in .uns to store result.
    **kwargs: Other parameters of sc.tl.score_genes.
    
    Return:
    ----------
    DataFrame of pathway score of each condition.

    Usage:
    ----------
    >>> terms = ['Wnt signaling pathway', 'nodal signaling pathway', 'fibroblast growth factor receptor signaling pathway']
    >>> gene_set = de.get_go_genes(terms)
    >>> score_matrix = de.score_genes(adata, condition = 'leiden', geneset = gene_set)
    '''
    if geneset is None:
        geneset = {'geneset':adata.var_names.tolist()}
    if isinstance(geneset, list):
        geneset = {'geneset':geneset}
    scores = []
    terms = []
    for x in geneset:
        print('Score genes for %s'%x)
        try:
            if method == 'mean':
                gs = np.array(geneset[x])[pd.Series(geneset[x]).isin(adata.var_names)] 
                scores.append(np.array(adata[:,gs].X.mean(1)).flatten())
            else:
                sc.tl.score_genes(adata, geneset[x], **kwargs)
                scores.append(adata.obs['score'].values)
            terms.append(x)
        except:
            pass
    if len(scores) > 1:
        df = pd.DataFrame(np.vstack(scores).T)
    else:
        df = pd.DataFrame(scores).T
    df.columns = terms
    df.index = adata.obs_names.tolist()
    if normalize:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, columns=df.columns, index = df.index)
    adata.uns[uns_keys] = df
    return(df)
    
def score_genes_matrix(adata, 
                       condition, 
                       method = 'mean',
                       score = None,
                       geneset = None,
                       normalize = True,
                       uns_key = 'score_matrix',
                       **kwargs):
    '''
    Scoring group of cells for pathway.
    
    Parameters:
    ----------
    adata: anndata.AnnData.
    condition: Condition in adata.obs for geneset comparison.
    method: mean of gene expression, or scanpy method.
    score: DataFrame of gene score calculated by de.score_genes.
    geneset: Dict of gene set.
    normalize: Bool. Normalize score of each pathway across all conditions.
    uns_key: The place in .uns to store result.
    **kwargs: Other parameters of sc.tl.score_genes.
    
    Return:
    ----------
    DataFrame of pathway score of each condition.

    Usage:
    ----------
    >>> terms = ['Wnt signaling pathway', 'nodal signaling pathway', 'fibroblast growth factor receptor signaling pathway']
    >>> gene_set = de.get_go_genes(terms)
    >>> score_matrix = de.score_genes_matrix(adata, condition = 'leiden', geneset = gene_set)
    '''
    if score is None:
        df = score_genes(adata, geneset)
    else:
        df = score
    df[condition] = adata.obs[condition].values
    ave_score = df.groupby(df[condition]).mean()
    if normalize:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(ave_score)
        ave_score = pd.DataFrame(scaled_data, columns=ave_score.columns, index = ave_score.index)
    adata.uns[uns_key] = ave_score.T
    return(ave_score.T)

def oneStepUmap(adata,features=None, n_neighbors=15, n_pcs=40, umap = True, resolution = 0.3, rescale = True):
    '''
    One-click generation of UMAP dimensionality reduction layout and Leiden clustering results for single-cell data

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (core object for single-cell data in Scanpy ecosystem), containing normalized gene expression data.
    features : list of str, optional
        List of gene names to use for PCA. 
        If None (default), I will calculate the hvg uses the genes present in adata.X.
    n_neighbors : int, optional
        Number of nearest neighbors to construct the k-nearest neighbor (kNN) graph for UMAP/clustering, default=15.
    n_pcs : int or None, optional
        Number of principal components (PCs) to retain for downstream kNN/UMAP/clustering, default=40.
        Uses fbpca (fast randomized PCA) to accelerate computation. If None, using adata.obsm['X_pca']
    umap : bool, optional
        Whether to perform UMAP dimensionality reduction, default=True.
        If True, UMAP coordinates are stored in adata.obsm['X_umap'].
    resolution : float or None, optional
        Resolution parameter for Leiden clustering (controls cluster granularity), default=0.3.
        Larger values result in more clusters; if None, skips Leiden clustering step.
    rescale : bool, optional
        Whether to rescale (standardize) features before PCA, default=True.
        Rescaling sets each feature to mean=0 and variance=1, critical for meaningful PCA on gene expression data.

    Returns
    -------
    anndata.AnnData
        Updated AnnData object with the following additions:
        - adata.obsm['X_pca']: PCA coordinates (if n_pcs is not None)
        - adata.uns['neighbors']: kNN graph metadata (if n_neighbors is not None)
        - adata.obsm['X_umap']: UMAP coordinates (if umap=True)
        - adata.obs['leiden']: Leiden cluster labels (if resolution is not None)

    Examples
    --------
    >>> import scanpy as sc
    >>> # Load example single-cell data
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> # Run one-step UMAP + clustering
    >>> adata = oneStepUmap(adata,n_neighbors=15,n_pcs=30,resolution=0.5)
    >>> sc.pl.umap(adata, color='leiden')
    '''
    if n_pcs is not None:
        adata = runPCA(adata,features=features,n_components=n_pcs,rescale=rescale)
    if n_neighbors is not None:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    if umap:
        sc.tl.umap(adata)
    if resolution is not None:
        sc.tl.leiden(adata, resolution = resolution)
    return(adata)





# def trajectoryPlot(adata,
#                    trajectory_ds,
#                    basis = 'X_umap',
#                    day_field = 'day',
#                    linestyle = 'solid',
#                    trajectory_label = 'label_latest',
#                    edgecolors = '#000000',
#                    color = 'time_label',
#                    color_map = 'plasma',
#                    linecolor = None,
#                    dotcolor = None,
#                    add_outline = True,
#                    size = 2,
#                    edgewidth = 2,
#                    point_size=1,
#                    line_alpha = 1,
#                    point_alpha = 1,
#                    ncol = 3,
#                    linewidths = 1,
#                    figsize = (8,8)):
    
#     ## 计算所有Lineage在每个时刻的中心的坐标
#     aggr_rep = []
#     days = np.sort(adata.obs[day_field].unique())
#     for t in days:
#         mask = adata.obs[day_field] == t
#         score = trajectory_ds[mask].X.T
#         score_sum = np.array(score.sum(1))
#         center_pos = np.array(np.dot(score, adata[mask].obsm[basis][:,:2]))
#         aggr_rep.append((center_pos.T/score_sum).T)
#     aggr_rep = np.vstack(aggr_rep)
#     mean_ann = anndata.AnnData(X = aggr_rep, obs = pd.DataFrame({'trajectory':np.tile(trajectory_ds.var_names, len(days)),
#                                                                  'day':np.repeat(days, trajectory_ds.shape[1])}))
    
#     ncol = min(trajectory_ds.shape[1], ncol)

#     trace_color = dict(zip(adata.obs[trajectory_label].cat.categories.tolist(), adata.uns[f'{trajectory_label}_colors']))
#     time_color = dict(zip(adata.obs[day_field].cat.categories.tolist(), adata.uns[f'{day_field}_colors']))

#     if trajectory_ds.shape[1] == ncol:
#         nc = ncol
#         nr = 1
#         pp = list(np.arange(ncol))
#         rm_pp = pp.copy()
#     elif trajectory_ds.shape[1] > ncol:
#         nc = ncol # number of columns
#         nr = np.ceil(trajectory_ds.shape[1]/nc).astype(int) # number of rows
#         pp = list(itertools.product(np.arange(nr), np.arange(nc)))
#         rm_pp = pp.copy()   
#     _, axs = plt.subplots(nr,nc, figsize = figsize)
#     for i in range(trajectory_ds.shape[1]):
#         if nc == nr == 1:
#             ax = axs
#             rm_pp = None
#         elif (nc == 1) | (nr ==1):
#             ax = axs[pp[i]]
#             rm_pp.pop(0)
#         else:
#             ax = axs[pp[i][0], pp[i][1]]
#             rm_pp.pop(0)
#         if linecolor is None:
#             linecolor_ax = trace_color[trajectory_ds.var_names[i]]
#         else:
#             linecolor_ax = linecolor
#         sub_mean_ann = mean_ann[mean_ann.obs['trajectory'] == trajectory_ds.var_names[i]] 

#         ## 绘制每个lineage在不同时间点之间的连线
#         ax.plot(sub_mean_ann.X[:,0], 
#                 sub_mean_ann.X[:,1],
#                 lw=linewidths,
#                 linestyle=linestyle,
#                 alpha = line_alpha,
#                 c=linecolor_ax,
#                 zorder = 2)
#         for t in days:
#             ## 令点的颜色为lineage的默认颜色
#             if dotcolor is None:
#                 dotcolor_ax = trace_color[trajectory_ds.var_names[i]]
#             elif dotcolor == 'time':
#                 dotcolor_ax = time_color[t]
#             else:
#                 dotcolor_ax = dotcolor
            
#             ## 绘制每个lineage在每个时间点的中心点
#             ax.scatter(sub_mean_ann.X[:,0][sub_mean_ann.obs[day_field]==t], 
#                        sub_mean_ann.X[:,1][sub_mean_ann.obs[day_field]==t], 
#                        alpha = point_alpha,
#                        c=dotcolor_ax,
#                        s=point_size,
#                        edgecolors = edgecolors,
#                        linewidth = edgewidth,
#                        zorder=3)

#         if color == 'score':
#             sc.pl.embedding(adata, basis = basis,
#                             color = trajectory_ds.var_names[i],title=trajectory_ds.var_names[i],
#                             size = size,
#                             add_outline = add_outline,
#                             ax = ax, color_map=color_map, frameon = False, show=False)
#         if color == 'label':
#             val = np.array(adata.obs[trajectory_label])
#             val[val != trajectory_ds.var_names[i]] = 'other'
#             sub_adata = adata[val=='other'].concatenate(adata[val==trajectory_ds.var_names[i]])
#             sub_adata.obs = pd.DataFrame(index=sub_adata.obs.index.values)
#             sub_adata.obs['trace_label'] = pd.Categorical(np.concatenate([val[val=='other'], val[val==trajectory_ds.var_names[i]]]),
#                                                           ['other',trajectory_ds.var_names[i]])

#             sub_adata.uns['trace_label_colors'] = ['#EBEBEB', trace_color[trajectory_ds.var_names[i]]]
#             sc.pl.embedding(sub_adata, basis = basis,
#                             color = 'trace_label', 
#                             add_outline = add_outline,
#                             title=trajectory_ds.var_names[i], 
#                             ax = ax, frameon = False, show=False)

#         if color == 'time_label':
#             # 把每个时间点分数最大的细胞群体挑选出来展示
#             df = adata.obs.loc[:, [day_field,trajectory_label,trajectory_ds.var_names[i]]]
#             df = df.groupby([day_field,trajectory_label]).mean().reset_index()
#             df = df.loc[df.groupby([day_field])[trajectory_ds.var_names[i]].idxmax()]
#             ## 在每个时间点中，将不属于某个特定器官的细胞的时间ID设为0（灰色表示），其它时间点为彩色
#             idx = []
#             for t in range(df.shape[0]):
#                 idx.append(adata.obs_names.values[(adata.obs[day_field] == df.iloc[t,:][0]) & (adata.obs[trajectory_label] != df.iloc[t,:][1])])
#             idx = np.concatenate(idx)
#             temp_labs = pd.Series(np.array(adata.obs[day_field]), index = adata.obs_names)
#             temp_labs[idx] = 0
            
#             ## 将轨迹分数为0的时间ID设为0
#             score = adata.obs[trajectory_ds.var_names[i]].copy()
#             zero_score_index = score.index.values[score == 0]
#             temp_labs[zero_score_index] = 0

#             ## 设置时间ID的颜色
#             adata.obs['temp_color'] = temp_labs
#             sub_adata = adata[temp_labs==0].concatenate(adata[temp_labs!=0])
#             sub_adata.obs['temp_color'] = pd.Categorical(sub_adata.obs['temp_color'])
            
#             ## 时间ID为0的细胞为灰色 #EBEBEB
#             sub_adata.uns['temp_color_colors'] = ['#EBEBEB'] + list(adata.uns[f'{day_field}_colors'])

#             ## 绘制UMAP
#             sc.pl.embedding(sub_adata, basis = basis,
#                             color = 'temp_color', 
#                             add_outline = add_outline,
#                             title=trajectory_ds.var_names[i], #legend_loc ='on data',
#                             ax=ax, frameon = False, show=False)
                            
#             ## 时间点太多了，这里设置不显示时间legend                
#             legend=ax.legend('')
#             legend.remove()

#     # remove unused subplots
#     if not rm_pp == None:
#         if len(rm_pp) > 0:
#             for rmplot in rm_pp:
#                 if nc == 1 | nr ==1:  
#                     axs[rmplot].set_axis_off()
#                 else:
#                     axs[rmplot[0], rmplot[1]].set_axis_off()   



def stackbarplot(adata,
                 x, y,
                 df = None,
                 y_colors = None, 
                 type = 'ratio',
                 sort = True,
                 width = 0.8,
                 linewidth = 0, 
                 legend_ncol = 2,
                 log_yaxis=False,
                 figsize = None, 
                 save=None):
    """
    Generate a stacked bar plot to visualize composition of categorical groups (y) across x-axis categories.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell data with categorical annotations in `.obs`.
    x : str
        Column name in `adata.obs` defining the x-axis categories (e.g., 'time_point', 'sample', 'condition').
        Must be a categorical variable.
    y : str
        Column name in `adata.obs` defining the stacked categories (e.g., 'cell_type', 'cluster', 'state').
        Must be a categorical variable.
    df : pandas.DataFrame, optional
        Precomputed DataFrame for stacked bar plot (rows = y categories, columns = x categories).
        If provided, skips calculation from `adata` and uses this data directly.
    y_colors : list or dict, optional
        Custom color palette for `y` categories:
        - List: Colors in order of `adata.obs[y].cat.categories`
        - Dict: Mapping of `y` category names to hex/RGB colors
        If None, uses existing palette from `adata.uns[f'{y}_colors']`.
    type : str, default='ratio'
        Type of values to plot on the y-axis:
        - 'ratio': Proportion (0 to 1) of each `y` category in each `x` group
        - 'count': Absolute number of cells for each `y` category in each `x` group
    sort : bool, default=True
        Whether to sort the stacked `y` categories:
        Sorts by (1) first occurrence (earliest `x` category where the `y` category exceeds 1% of the group)
        and (2) maximum proportion/count across all `x` categories.
    width : float, default=0.8
        Width of the bars (0 to 1, where 1 = full width of x-axis ticks).
    linewidth : float, default=0
        Width of bar outline lines (0 = no outline).
    legend_ncol : int, default=2
        Number of columns in the legend (for compact layout).
    log_yaxis : bool, default=False
        Whether to apply log scale to the y-axis (useful for count data with large dynamic range).
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, uses matplotlib default.
    save : str, optional
        File path to save the plot (e.g., 'stacked_barplot.pdf'). If None, plot is not saved.

    Returns
    -------
    pandas.DataFrame
        Transposed stacked data (rows = x categories, columns = y categories) used for plotting.
        Values are either proportions (0-1) or counts, matching the `type` parameter.

    Raises
    ------
    ValueError
        If `type` is not 'ratio' or 'count'.
    KeyError
        If `x` or `y` are not present in `adata.obs.columns`.
        If `y_colors` is a dict but missing keys for some `y` categories.

    Notes
    -----
    - Both `x` and `y` must be categorical variables in `adata.obs` (use `pd.Categorical` if needed).
    - When `df=None`, the function preserves the order of `x` categories from `adata.obs[x].cat.categories`.
    - The legend is positioned outside the plot (top-right) to avoid overlapping with bars.
    - Plot styling removes top/right spines and optimizes tick label font size for readability.

    Examples
    --------
    >>> # Basic proportion plot (cell type composition across time points)
    >>> ratio_df = stackbarplot(
    ...     adata,
    ...     x='time_point',
    ...     y='cell_type',
    ...     type='ratio',
    ...     figsize=(10, 6),
    ...     save='cell_type_time_proportion.pdf'
    ... )

    >>> # Count plot with custom colors and log scale
    >>> color_map = {'T_cell': '#FF5733', 'B_cell': '#33FF57', 'Macrophage': '#3357FF'}
    >>> count_df = stackbarplot(
    ...     adata,
    ...     x='sample',
    ...     y='cell_type',
    ...     y_colors=color_map,
    ...     type='count',
    ...     log_yaxis=True,
    ...     legend_ncol=3
    ... )

    >>> # Use precomputed DataFrame
    >>> precomputed_df = pd.DataFrame(...)  # Rows = cell types, Columns = time points
    >>> stackbarplot(adata, x='time_point', y='cell_type', df=precomputed_df)
    """
    obs = adata.obs
    if df is not None:
        stack_ratio = df.T
        ymax = stack_ratio.sum(0).max()
    else:
        stack_ratio = []
        if type == 'count':
            ymax = None
            for t in obs[x].cat.categories:
                sub_obs = obs[obs[x] == t]
                ratio = (sub_obs[y].value_counts())[obs[y].cat.categories.tolist()]
                stack_ratio.append(ratio.tolist())
        else:
            ymax = 1
            for t in obs[x].cat.categories:
                sub_obs = obs[obs[x] == t]
                ratio = (sub_obs[y].value_counts()/sub_obs.shape[0])[obs[y].cat.categories.tolist()]
                stack_ratio.append(ratio.tolist())
        stack_ratio = pd.DataFrame(np.array(stack_ratio).T,
                                index = obs[y].cat.categories.tolist(),
                                columns = obs[x].cat.categories.tolist())
        if sort:
            first_time = stack_ratio.apply(lambda x:min(np.where(x>max(x)*0.01)[0]),axis = 1).values # 根据首次出现的时间排序
            max_ratio = stack_ratio.apply(max,axis = 1).values # 根据占比排序
            _, _, new_index = zip(*sorted(zip(first_time, max_ratio, stack_ratio.index.values)))

            stack_ratio = stack_ratio.loc[new_index,:]

    labels= stack_ratio.index.values

    
    if y_colors is None:
        colors = adata.uns['%s_colors'%y]
    else:
        colors = y_colors
    color_dict = dict(zip(obs[y].cat.categories.tolist(),colors))
    fig, ax = plt.subplots(figsize = figsize)
    for i in range(stack_ratio.shape[0]):
        ax.bar(stack_ratio.columns.values,
               stack_ratio.iloc[i,:].values,label=labels[i],
               color = color_dict[labels[i]],
               bottom=stack_ratio.iloc[:i,:].sum(0).values,
               linewidth = linewidth,
               width = width)
    ax.grid(False)
    ax.margins(x=0.02)
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(90)
        label.set_fontsize(8)
    for label in ax.get_yticklabels():
        label.set_fontsize(8) 
    ax.legend(bbox_to_anchor=(1.1, 1.05),ncol=legend_ncol, frameon=False)
    ax.set_ylim(0,ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False,left=False, labelleft=True)
    ax.tick_params(axis='y', which='both', bottom=False, top=False,left=True, labelleft=True)
    if log_yaxis:
        ax.set_yscale('log') 
    if save is not None:
        plt.savefig(save)
    return(stack_ratio.T)

# def dotGeneExp(adata, x, y, 
#                gene,
#                cmap = 'YlGnBu_r',
#                x_order = None,
#                y_order = None,
#                dot_min = 0.05,
#                dot_max = 1,
#                bbox_to_anchor=(1.20, 0.8),
#                figsize = (5.5,3.5),
#                verbose = False):
#     x_name = []
#     y_name = []
#     prop = []
#     mean = []
#     if x_order is None:
#         x_list = pd.Categorical(adata.obs[x]).categories
#     else:
#         x_list = x_order
#     if y_order is None:
#         y_list = pd.Categorical(adata.obs[y]).categories
#     else:
#         y_list = y_order
#     adata = adata[:,gene].copy()
#     for xi in x_list:
#         if verbose:
#             print(xi)
#         for yi in y_list:
#             sub_adata = adata[(adata.obs[x] == xi) & (adata.obs[y] == yi),]
#             if sub_adata.shape[0] > 0:
#                 prop.append((nonzero(sub_adata.X)/sub_adata.X.shape[0])[0])
#                 mean.append(sub_adata.X.mean())

#             else:
#                 prop.append(0)
#                 mean.append(0)
#             x_name.append(xi)
#             y_name.append(yi)
#     df = pd.DataFrame({'x':pd.Categorical(x_name, x_list),
#                        'y':pd.Categorical(y_name, y_list), 
#                        'prop':prop, 'mean':mean})

#     prop_plot = df['prop'].copy()
#     prop_plot[prop_plot <= dot_min] = 0
#     prop_plot[prop_plot >= dot_max] = dot_max

#     fig, (ax, cax) = plt.subplots(nrows=2,figsize=figsize, 
#                       gridspec_kw={"height_ratios":[1, 0.05]})
#     fig.tight_layout(pad=2.0)

#     scatter = ax.scatter(df['x'], df['y'], cmap = cmap,
#                           s = prop_plot*100,c = df['mean'],
#                           edgecolors = '#000000',
#                 linewidths = 1,label = 'Scn7a')

#     handles, labels = scatter.legend_elements(prop="sizes",num=5, alpha=0.6)
#     legend = ax.legend(handles, labels, loc="upper right", 
#                         title="Proportion",bbox_to_anchor=bbox_to_anchor,
#                         fancybox=False, shadow=False)
    
#     cbar = fig.colorbar(scatter, cax = cax, orientation='horizontal')
#     for label in ax.get_xticklabels():
#         label.set_ha("center")
#         label.set_rotation(90)
#     ax.margins(x=0.02)
#     ax.set_title(gene)


def fateMapping(adata,
                predict_adata=None, 
                predict_labels=None, 
                batch_key_dict=None,
                features = None,
                condition = None,
                plot_mapping_layout=False,
                plot_mapping_ratio=False,
                translate_gene_names = False,
                merge_dup_genes = False,
                save_raw_exprs = True,
                layout='X_umap',
                runpca = True,
                runcca = False,
                runharmony = True,
                pca_rescale = True,
                n_components = 30,
                upper_gene_name=False,
                r = 0.1,
                figsize=(3, 8),
                annot = False,
                save = False,
                **kwarg):
    '''
    Use harmony to integrate two datasets, predict labels and mapping layout.
    ------
    
    adata: anndata.AnnData base data for mapping. 
    predict_adata: anndata.AnnData predcit data. If predict_adata is None, it means adata is the integreted data.
    predict_labels: str. Predict labels in adata.obs for predict_adata.
    batch_key_dict: Dict with key in adata.obs for batch labels.
    features: Genes used for integration.
    condition: key in predict_adata.obs for comparison with predict_labels in adata
    plot_mapping_layout: Bool. Plot cell of predict_adata on adata.
    plot_mapping_ratio: Bool. Plot mapping ratio of each group in predict_adata.obs[condition] to adata.obs[predict_labels]
    translate_gene_names: Traslate to human gene symbols.
    merge_dup_genes: Bool. Merge duplicated gene as mean after translate_gene_names.
    save_raw_exprs: Bool. Save raw data after concatenate two data.
    layout: key in adata.obsm for plotting mapping result.
    runpca: Bool. Whether to run pca.
    runcca: Bool. Whether to run cca. If runcca is True, I will perform cca based on the pca result. Else, cca will be performed used HVGs.
    runharmony: Bool. Whether to run harmony to integrate two dataset.
    n_components: Number of components for calculating pca or cca.
    upper_gene_name: Upper gene names.
    r: Parameter to modify cell order when plot_mapping_ratio is True.
    figsize: Figure size.
    annot: Seaborn heatmap parameter for plot_mapping_ratio.
    save: Bool. If True, save figures automatically.
    kwarg: Other parameters for sc.pl.embedding.

    '''
    if batch_key_dict is None:
        batch_key_dict = {'fatemap':['base','predict']}

    key = list(batch_key_dict.keys())[0]
    raw_coords = None
    if predict_adata is not None:
        raw_coords = adata.obsm
        adata.obs[key] = batch_key_dict[key][0]
        predict_adata.obs[key] = batch_key_dict[key][1]
        adata.obs_names_make_unique()
        predict_adata.obs_names_make_unique()

        if save_raw_exprs:
            adata_raw = adata.concatenate(predict_adata, join = 'outer')
        else:
            adata_raw = None

        if translate_gene_names:
            # Upper gene symbols and match human gene symbols.
            adata = symbolMouse2Human(adata,predict_adata, merge_dup_genes = merge_dup_genes)
            adata.var_names_make_unique()
        merge_data = adata.concatenate(predict_adata, join = 'inner')
        merge_data.raw = adata_raw

        if features is not None:
            features = np.array(features)[pd.Series(features).isin(merge_data.var_names)]
        # 批次矫正
        if runpca:
        # if method == 'pca':
            print('Runing PCA...')
            merge_data = oneStepUmap(merge_data,features=features,n_pcs=n_components,umap = False, resolution = None, rescale=pca_rescale)
            basis = 'X_pca'
        if runcca:
            if runpca:
                use_rep = 'X_pca'
            else:
                use_rep = None
        # if method == 'cca':
            print('Runing CCA...')
            merge_data = runCCA(merge_data, merge_data.obs[key]==batch_key_dict[key][0],
                                merge_data.obs[key]==batch_key_dict[key][1], use_rep=use_rep,
                                n_components=n_components, features=features) #20
            basis = 'X_cca'
        if runharmony:
            sce.pp.harmony_integrate(merge_data, key, basis = basis, adjusted_basis = '%s_harmony'%basis)
            basis = '%s_harmony'%basis
        merge_data.uns['fatemap_basis'] = basis
    else:
        merge_data = adata

    batch0 = merge_data[merge_data.obs[key] == batch_key_dict[key][0]]
    batch1 = merge_data[merge_data.obs[key] == batch_key_dict[key][1]]

    if predict_adata is not None:
        batch0.obs_names = adata.obs_names
        batch1.obs_names = predict_adata.obs_names

    batch0_pcs = pd.DataFrame(batch0.obsm[merge_data.uns['fatemap_basis']], index = batch0.obs_names)
    batch1_pcs = pd.DataFrame(batch1.obsm[merge_data.uns['fatemap_basis']], index = batch1.obs_names)

    mapping = fastKnn(batch0_pcs, batch1_pcs,n_neighbors=1)
    mapping = pd.Series([batch0_pcs.index[i] for i in mapping[1][:,0]], index = batch1_pcs.index.values)

    if predict_labels is not None:
        if not isinstance(predict_labels,list):
            predict_labels = [predict_labels]
        anno = batch0.obs.loc[mapping.values,predict_labels]
        for label in predict_labels:
            if predict_adata is not None:
                if anno[label].dtype == 'category':
                    predict_adata.obs[label] = pd.Categorical(np.array(anno[label]),anno[label].cat.categories.tolist())
                else:
                    predict_adata.obs[label] = np.array(anno[label])
                if '%s_colors' in adata.uns:
                    predict_adata.uns['%s_colors'] = adata.uns['%s_colors']

    # 给predict_adata赋予adata的所有layout
    if raw_coords is not None:
        ids = mapping[batch1.obs_names].values
        for ll in raw_coords:
            coord = pd.DataFrame(np.array(raw_coords[ll]), index = batch0.obs_names.values)
            predict_layout = coord.loc[ids,:].values
            merge_data.obsm['%s'%ll] = np.vstack([coord, predict_layout])
            batch0.obsm[ll] = coord
            batch1.obsm[ll] = predict_layout
            if predict_adata is not None:
                predict_adata.obsm['%s_predict'%ll] = predict_layout
        
    # Plotting
    if plot_mapping_layout:
        if condition is not None:
            coord = pd.DataFrame(batch1.obsm[layout], index = batch1.obs_names.values)
            for cluster in np.unique(np.array(batch1.obs[condition])):
                bcs = batch1.obs_names[batch1.obs[condition] == cluster]
                # ids = mapping[bcs].values
                label = pd.Categorical(np.concatenate([np.repeat('Others', batch0.shape[0]), np.repeat(cluster, len(bcs))]),['Others',cluster])
                ps_adata = anndata.AnnData(obs = pd.DataFrame({'label':label},index = np.concatenate([batch0.obs_names.values, bcs])),
                                        obsm = {layout:np.vstack([np.array(batch0.obsm[layout]), coord.loc[bcs,:].values])})
                ps_adata.uns['label_colors'] = ['#E5E5E5','#2B7CD3']
                if save:
                    sc.pl.embedding(ps_adata, basis = layout, color = ['label'], title = cluster, frameon=False, save = '%s_mapping.pdf'%cluster,**kwarg)
                else:
                    sc.pl.embedding(ps_adata, basis = layout, color = ['label'], title = cluster, frameon=False,**kwarg)
    if plot_mapping_ratio:
        if condition is not None:
            stat = pd.DataFrame({'predict_labels':np.array(batch0.obs.loc[mapping.values,predict_labels[0]]),
                     'condition':batch1.obs.loc[mapping.index.values,condition]})
            count = pd.pivot_table(stat, index='predict_labels', columns='condition', aggfunc=len, fill_value=0)
            ratio = count.div(count.sum(axis=0), axis=1)

            # 根据熵值从小到大对数据框的列排序
            entropies = ratio.apply(entropy, axis=0, base=2)
            ratio = ratio[entropies.sort_values().index]
            order_row = np.argsort(ratio.sum(1).values)[::-1]
            ratio = ratio.iloc[order_row, :]
            col_dicts = {}
            exit_vals=[]
            for x in range(ratio.shape[1]):
                t0 = ratio.columns[x]
                if x < ratio.shape[1]-1:
                    t1 = ratio.columns[x+1]
                else:
                    t1 = ratio.columns[x]
                df0 = ratio.loc[~ratio.index.isin(exit_vals),:]
                sort_t0 = df0[t0].sort_values(ascending = False)
                df1 = ratio.loc[~ratio.index.isin(exit_vals),:]
                sort_t1 = df1[t1].sort_values(ascending = False)
                # import pdb;pdb.set_trace()
                if len(sort_t1) > 1:
                    names = sort_t0[(sort_t0>r*sort_t0[0])&(sort_t0>r*4*sort_t1[1])].index.values
                elif len(sort_t0) > 1:
                    names = sort_t0[sort_t0>r*sort_t0[0]].index.values
                else:
                    names = sort_t0.index.values
                exit_vals.extend(names)
                col_dicts[t0] = []
                col_dicts[t0].extend(names)
            col_dicts['others'] = list(ratio.index.values[~ratio.index.isin(exit_vals)])
            row_names = np.concatenate([col_dicts[i] for i in col_dicts])
            df = ratio.loc[row_names,:]
            merge_data.uns['mapping_ratio'] = df
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(df, cmap = mymap_wr, annot=annot, fmt=".1",xticklabels=True, yticklabels=True,linewidths=1, linecolor='black', clip_on=False,ax=ax)
            if save:
                plt.savefig('figures/%s_mapping_ratio.pdf'%condition)
            else:
                plt.show()
    if predict_adata is not None:
        return(merge_data)


# def compareSpeciesStage(adata,
#                         stage_label, 
#                         condition_label,
#                         condition1, 
#                         condition2, 
#                         basis = 'X_cca',
#                         metric = 'similarity',
#                         n_bins1=None,
#                         n_bins2=None,
#                         num=50,
#                         cmap = 'RdYlBu_r',
#                         figsize = (4.5,4),
#                         save = None,
#                         **kwargs):
    
#     if condition1 is None:
#         condition1 = np.unique(adata.obs[condition_label])[0]
#     if condition2 is None:
#         condition2 = np.unique(adata.obs[condition_label])[1]
#     conditions = [condition1, condition2]
#     mean_rep1 = []
#     mean_rep2 = []
#     # Mean of condition 1
#     sub_adata = adata[adata.obs[condition_label] == condition1]

#     if n_bins1 is None:
#         n_bins = len(adata.obs[stage_label].unique())
#     else:
#         n_bins = n_bins1
#     cut_bins1 = pd.cut(np.array(sub_adata.obs[stage_label]), n_bins, labels=False)

#     sub_adata.obs[stage_label] = np.array(sub_adata.obs[stage_label])
#     sub_adata.obs['bin']=cut_bins1
#     x1 = sub_adata.obs.groupby('bin')[stage_label].mean().values

#     for t in np.sort(np.unique(cut_bins1)):
#         mean_rep1.append(np.array(sub_adata.obsm[basis][cut_bins1==t].mean(0)).flatten())


#     # Mean of condition 2
#     sub_adata = adata[adata.obs[condition_label] == condition2]

#     if n_bins2 is None:
#         n_bins = len(adata.obs[stage_label].unique())
#     else:
#         n_bins = n_bins2
#     cut_bins2 = pd.cut(np.array(sub_adata.obs[stage_label]), n_bins, labels=False)

#     sub_adata.obs[stage_label] = np.array(sub_adata.obs[stage_label])
#     sub_adata.obs['bin']=cut_bins2
#     x2 = sub_adata.obs.groupby('bin')[stage_label].mean().values

#     for t in np.sort(np.unique(cut_bins2)):
#         mean_rep2.append(np.array(sub_adata.obsm[basis][cut_bins2==t].mean(0)).flatten())
#     cormat = fast_corrcoef(np.vstack(mean_rep1), np.vstack(mean_rep2))
    
    
#     # 平滑矩阵
#     x1_new = np.linspace(min(x1), max(x1), num)
#     x2_new = np.linspace(min(x2), max(x2), num)

#     # 创建插值函数
#     interp_func = interpolate.interp2d(x2, x1,cormat, kind='linear') # linear, cubic, quintic

#     # 进行插值处理
#     cormat_interp = pd.DataFrame(interp_func(x2_new, x1_new), index = np.round(x1_new,2), columns = np.round(x2_new,2))
#     fig, ax = plt.subplots(1, 1,figsize = figsize)
#     if metric == 'similarity':
#         ax = sns.heatmap(cormat_interp,cmap = cmap,ax=ax,**kwargs)
#     if metric == 'distance':
#         ax = sns.heatmap(1-cormat_interp,cmap = cmap,ax=ax,**kwargs)
#     ax.set_ylabel(condition1)
#     ax.set_xlabel(condition2)
#     ax.plot([0, len(x1_new)], [0, len(x2_new)], color='black', linewidth=1, linestyle='--')
#     if save is not None:
#         plt.savefig(save)
#     else:
#         plt.show()


# def symbolMouse2Human(adata_mouse, 
#                       adata_human,
#                       merge_dup_genes = True,
#                       anno_path = '/data1/home/jkchen/hlchen/0331/project/zhufeng_lung_stromal/HOM_MouseHumanSequence.rpt.txt'):
#     '''
#     Tranlate mouse gene symbol to human gene symbol
#     '''
#     mouse = adata_mouse.copy()
#     mouse.var['Raw_symbol'] = mouse.var_names
#     translate = pd.read_table(anno_path)
#     df = translate.loc[:,['DB Class Key','Common Organism Name','NCBI Taxon ID','Symbol']]
#     df.Symbol = [i.upper() for i in df.Symbol]
#     mouse.var_names = [i.upper() for i in mouse.var_names]

#     mouse.obs_names_make_unique()
#     adata_human.obs_names_make_unique()

#     mouse.var_names_make_unique()
#     adata_human.var_names_make_unique()

#     common_genes = np.intersect1d(mouse.var_names, adata_human.var_names)
#     # part1_mouse = mouse[:,common_genes]
#     part2_mouse = mouse[:,list(set(mouse.var_names)-set(common_genes))]
    
#     df_split = split(df, df['NCBI Taxon ID'])
#     df_split[10090] = df_split[10090][df_split[10090]['Symbol'].isin(part2_mouse.var_names)]
#     df = pd.concat([df_split[9606],df_split[10090]])
#     id_split = split(df, df['DB Class Key'])

#     m_genes = []
#     h_genes = []
#     for idx in id_split:
#         gene_split = split(id_split[idx]['Symbol'], id_split[idx]['NCBI Taxon ID'])
#         # 条件1 当一个或多个人的基因对应零个鼠的基因时，去除这些基因
#         if (10090 in list(gene_split.keys())) & (9606 in list(gene_split.keys())):

#             if len(gene_split[9606]) == 1:
#                 if len(gene_split[10090]) == 1:
#                     m_genes.append(gene_split[10090][0])
#                     h_genes.append(gene_split[9606][0])
#                 # else:
#                 #     # 条件2 当一个人的基因对应多个鼠的基因时，以人的基因命名，鼠的基因取平均
#                 #     gexp.append(part2_mouse[:,gene_split[10090]])
#                 #     print(part2_mouse[:,gene_split[10090]])
#             else:
#                 # 条件3 当多个人的基因对应一个鼠的基因时，重复鼠的基因以与人的基因数量对应
#                 if len(gene_split[10090]) == 1:
#                     m_genes.extend(gene_split[10090]*len(gene_split[9606]))
#                     h_genes.extend(gene_split[9606])

#                 # 条件4 当多个人的基因对应多个鼠的基因时，只保留共享基因名的基因
#                 if len(gene_split[10090]) > 1:
#                     mouse_genes = np.array(gene_split[10090])[pd.Series([i.upper() for i in gene_split[10090]]).isin(gene_split[9606])]
#                     if len(mouse_genes) > 0:
#                         m_genes.extend(list(mouse_genes))
#     mouse = mouse[:,np.concatenate([common_genes,m_genes])]
#     mouse.var_names = np.concatenate([common_genes,h_genes])
#     mouse = mouse[:,~mouse.var.index.duplicated()]
    
#     if merge_dup_genes:
#         print('Merge gene expression with common symbols...')
#         count = mouse.var_names.value_counts()
#         # import pdb;pdb.set_trace()
#         data1 = mouse[:,mouse.var_names.isin(count[count==1].index.values)]
#         gs = count[count>1].index.values
#         exp = []
#         for g in gs:
#             exp.append(csr_matrix(mouse[:,g].X.mean(1)))
#         exp = sp.hstack(exp)
#         var = mouse.var
#         var = var[~var.index.duplicated()]
#         X = sp.hstack([data1.X, exp])
#         mouse = anndata.AnnData(X = X, obs = mouse.obs, var = var.loc[np.concatenate([data1.var_names.values,gs]),:],
#                         obsm = mouse.obsm, uns = mouse.uns)
    
#     return(mouse)

def label_cells(adata, 
                predicted_adata=None,
                annotation=None,
                key='leiden',
                label = 'celltype', 
                default_label = 'Undefined'):
    '''
    Assign cell type labels to AnnData objects, either via pre-defined cluster annotations 
    or by migrating existing labels between overlapping cell populations in two datasets.

    This function supports two core workflows:
    1. Label cells in `adata` using a cluster-to-celltype mapping dictionary (`annotation`).
    2. Migrate pre-existing cell labels from `adata` to `predicted_adata` (for overlapping cell IDs).
    Labels are added/updated in the `obs` attribute of the AnnData object(s).

    Parameters
    ----------
    adata : anndata.AnnData
        Primary AnnData object to assign labels to (contains cell clusters/labels in `obs`).
    predicted_adata : anndata.AnnData or None, optional
        Secondary AnnData object to migrate labels to (requires overlapping cell IDs with `adata`).
        If None (default), only `adata` is modified.
    annotation : dict or None, optional
        Dictionary mapping cluster IDs (keys) to cell type labels (values) (e.g., {'0': 'T cell', '1': 'B cell'}).
        - If provided: Assign labels to `adata` (and `predicted_adata` if specified) based on cluster IDs in `key`.
        - If None: Migrate existing `label` column from `adata` to `predicted_adata` (for overlapping cells).
    key : str, optional
        Column name in `adata.obs` containing cluster IDs (e.g., cluster labels), default='leiden'.
        Only used when `annotation` is provided (to map clusters to cell types).
    label : str, optional
        Column name in `obs` to store the cell type labels, default='celltype'.
        - If the column exists: Update values in-place.
        - If the column does not exist: Create a new column with `default_label` as initial value.
    default_label : str, optional
        Default label assigned to cells when creating a new `label` column (before custom annotations), default='Undefined'.
        Only applies when the `label` column does not exist in `adata`/`predicted_adata`.

    Returns
    -------
    None
        The function modifies `adata` (and `predicted_adata` if specified) in-place; no return value.

    Examples
    --------
    >>> # Example 1: Assign cell type labels via cluster annotation
    >>> import scanpy as sc
    >>> import devendo as de
    >>> adata = sc.read_h5ad('single_cell_data.h5ad')
    >>> celltype_annotation = {'0': 'CD4+ T cell', '1': 'CD8+ T cell', '2': 'NK cell'}
    >>> de.label_cells(
    ...     adata=adata,
    ...     annotation=celltype_annotation,
    ...     key='leiden',
    ...     label='celltype',
    ...     default_label='Unknown'
    ... )

    >>> # Example 2: Migrate labels from adata to predicted_adata (overlapping cells)
    >>> predicted_adata = sc.read_h5ad('predicted_data.h5ad')
    >>> de.label_cells(
    ...     adata=adata,
    ...     predicted_adata=predicted_adata,
    ...     annotation=None,  # Use existing labels in adata.obs['celltype']
    ...     label='celltype'
    ... )
    '''
    if annotation is None:
        if label in adata.obs.columns:
            obs_names = np.intersect1d(predicted_adata.obs_names, adata.obs_names)
            if label not in predicted_adata.obs.columns:
                predicted_adata.obs[label] = default_label
            predicted_adata.obs[label] = np.array(predicted_adata.obs[label])
            predicted_adata.obs.loc[obs_names,label] = np.array(adata.obs.loc[obs_names,label])
        else:
            print('label is not in adata.obs')
    else:
        if label not in adata.obs.columns:
            adata.obs[label] = default_label

        if predicted_adata is not None:
            if label not in predicted_adata.obs.columns:
                predicted_adata.obs[label] = default_label

        for s in annotation:
            obs_names = adata.obs_names[adata.obs[key]==s]
            adata.obs[label] = np.array(adata.obs[label])
            adata.obs.loc[obs_names,label] = annotation[s]

            if predicted_adata is not None:
                obs_names = np.intersect1d(predicted_adata.obs_names, obs_names)
                predicted_adata.obs[label] = np.array(predicted_adata.obs[label])
                predicted_adata.obs.loc[obs_names,label] = annotation[s]




def predict_cell_labels(training_adata, 
                        predict_adata, 
                        features=None,
                        model = None,
                        training_label='organ',
                        predict_label='predict_label',
                        return_model = False,
                        cells_per_group=1000,
                        n_estimators = 100,
                        max_depth = 20,
                        min_samples_leaf = 2,
                        n_jobs = -1):
    """
    Predict cell labels using Random Forest classifier for single-cell RNA-seq data
    
    This function trains a Random Forest model on annotated single-cell data (training_adata) 
    and predicts cell labels for unannotated data (predict_adata). It handles feature gene 
    alignment between training and prediction datasets, and stores prediction results (labels/probabilities) 
    in the AnnData object for downstream analysis.

    Parameters
    ----------
    training_adata : anndata.AnnData
        Annotated AnnData object for model training, containing known cell labels in `obs` attribute.
    predict_adata : anndata.AnnData
        Unannotated AnnData object to predict cell labels for. Prediction results will be stored in this object.
    features : list of str or None, optional
        List of feature genes to use for model training/prediction. If None (default), uses all genes in training_adata.
    model : sklearn.base.ClassifierMixin or None, optional
        Pre-trained scikit-learn classifier model (must support fit/predict/predict_proba methods). 
        If None (default), a new Random Forest classifier is trained from scratch.
    training_label : str, optional
        Column name in `training_adata.obs` containing ground-truth cell labels (e.g., cell type/organ), default='organ'.
    predict_label : str, optional
        Column name in `predict_adata.obs` to store predicted cell labels, default='predict_label'.
    return_model : bool, optional
        Whether to return the trained Random Forest model along with predictions, default=False.
    cells_per_group : int, optional
        Maximum number of cells sampled per label group in training data (to balance class distribution), default=1000.
    n_estimators : int, optional
        Number of decision trees in the Random Forest classifier, default=100.
    max_depth : int, optional
        Maximum depth of each decision tree in the Random Forest. Deeper trees may cause overfitting, 
        while shallow trees may lead to underfitting, default=20.
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node (prevents overfitting), default=2.
    n_jobs : int, optional
        Number of parallel computing threads for model training/prediction. 
        -1 means using all available CPU cores, default=-1.

    Returns
    -------
    numpy.ndarray
        y_pred: Array of predicted cell labels for predict_adata (shape = [n_cells, ]). The results are 
    sklearn.ensemble.RandomForestClassifier, optional
        Trained Random Forest model (returned only if return_model=True)

    Examples
    --------
    >>> training_adata = sc.read_h5ad('annotated_sc_data.h5ad')
    >>> predict_adata = sc.read_h5ad('unannotated_sc_data.h5ad')
    >>> # Use highly variable genes as features
    >>> features = training_adata.var_names[training_adata.var['highly_variable']].tolist()
    >>> y_pred = predict_cell_labels(
    ...     training_adata=training_adata,
    ...     predict_adata=predict_adata,
    ...     training_label='cell_type',
    ...     features=features,
    ...     n_estimators=200,
    ...     max_depth=15,
    ...     n_jobs=8
    ... )

    """
    adata_train = choiceGroupCells(training_adata, training_label, cells_per_group=cells_per_group, seed = 123)

    if features is None:
        features = training_adata.var_names.tolist()
    used_features = np.array(features)[pd.Series(features).isin(training_adata.var_names)]
    X_train = training_adata[:,used_features].X
    y_train = training_adata.obs[training_label].values

    # 基因
    common_test_genes = list(set(predict_adata.var_names) & set(used_features))
    diff_test_genes = list(set(used_features) - set(predict_adata.var_names))
    
    common_test_genes_X = predict_adata[:,common_test_genes].X
    diff_test_genes_X = sparse.csr_matrix((predict_adata.shape[0], len(diff_test_genes)))
    pre_test_X = sparse.hstack([common_test_genes_X,diff_test_genes_X])
    pre_test_adata = anndata.AnnData(pre_test_X, var = pd.DataFrame(index = np.concatenate([list(common_test_genes),list(diff_test_genes)])))
    X_test = pre_test_adata[:,used_features].X
    if model is None:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            random_state=42,
            n_jobs=n_jobs
        )
        rf.fit(X_train, y_train)
    else:
        rf = model
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    predict_adata.obs['predict_label'] = y_pred
    predict_adata.uns['predict_proba'] = pd.DataFrame(y_pred_proba,index = predict_adata.obs_names.tolist(), columns=rf.classes_) # , index = predict_adata.obs_names.tolist(), columns=rf.classes_
    print("Results are save in .obs['predict_label'] and predict_adata.uns['predict_proba']")
    if return_model:
        return(y_pred, rf)
    else:
        return(y_pred)

def compareSpeciesStage(adata,
                        stage_label, 
                        condition_label,
                        condition1, 
                        condition2, 
                        basis = 'X_pca',
                        metric = 'similarity',
                        n_bins1=None,
                        n_bins2=None,
                        title = None,
                        num=50,
                        cmap = 'RdYlBu_r',
                        figsize = (4.5,4),
                        ax = None,
                        save = None,
                        **kwargs):

    """
    Compare developmental stage similarity between two conditions using dimensionality-reduced embeddings
    and visualize results as a heatmap with optional interpolation.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell data with embeddings in .obsm and metadata in .obs
    stage_label : str
        Column name in adata.obs containing developmental stage/ timepoint information (numeric values)
    condition_label : str
        Column name in adata.obs containing condition group labels (e.g., species, treatment, genotype)
    condition1 : str or None
        First condition to compare (e.g., 'SpeciesA'). If None, uses first unique value in condition_label
    condition2 : str or None
        Second condition to compare (e.g., 'SpeciesB'). If None, uses second unique value in condition_label
    basis : str, default='X_cca'
        Key in adata.obsm containing dimensionality-reduced embeddings (e.g., PCA, CCA, UMAP coordinates)
    metric : str, default='similarity'
        Metric for heatmap visualization:
        - 'similarity': Use correlation coefficients directly
        - 'distance': Use 1 - correlation coefficients (dissimilarity)
    n_bins1 : int or None, default=None
        Number of bins to discretize stage_label for condition1. If None, uses number of unique stages
    n_bins2 : int or None, default=None
        Number of bins to discretize stage_label for condition2. If None, uses number of unique stages
    title : str or None, default=None
        Title for the heatmap plot. If None, no title is displayed
    num : int, default=50
        Number of interpolation points for smoothing the correlation matrix (higher = smoother)
    cmap : str, default='RdYlBu_r'
        Colormap name for seaborn heatmap (matplotlib colormap format)
    figsize : tuple, default=(4.5,4)
        Figure size (width, height) in inches (only used if ax=None)
    ax : matplotlib.axes.Axes or None, default=None
        Pre-existing axes object to plot on. If None, creates new axes
    save : str or None, default=None
        File path to save the plot (e.g., 'stage_comparison.png'). If None, plot is not saved
    **kwargs : dict
        Additional keyword arguments passed to seaborn.heatmap (e.g., vmin, vmax, annot, fmt)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the heatmap plot

    Notes
    -----
    1. The function bins developmental stages for each condition, calculates mean embeddings per bin,
       then computes pairwise correlation between bins of the two conditions
    2. Interpolation is applied to create a smooth heatmap between original bin points
    3. A diagonal dashed line is added to highlight perfect stage matching
    """

    if condition1 is None:
        condition1 = np.unique(adata.obs[condition_label])[0]
    if condition2 is None:
        condition2 = np.unique(adata.obs[condition_label])[1]
    conditions = [condition1, condition2]
    mean_rep1 = []
    mean_rep2 = []
    # Mean of condition 1
    sub_adata = adata[adata.obs[condition_label] == condition1]

    if n_bins1 is None:
        n_bins = len(adata.obs[stage_label].unique())
    else:
        n_bins = n_bins1
    cut_bins1 = pd.cut(np.array(sub_adata.obs[stage_label]), n_bins, labels=False)

    sub_adata.obs[stage_label] = np.array(sub_adata.obs[stage_label])
    sub_adata.obs['bin']=cut_bins1
    x1 = sub_adata.obs.groupby('bin')[stage_label].mean().values

    for t in np.sort(np.unique(cut_bins1)):
        mean_rep1.append(np.array(sub_adata.obsm[basis][cut_bins1==t].mean(0)).flatten())


    # Mean of condition 2
    sub_adata = adata[adata.obs[condition_label] == condition2]

    if n_bins2 is None:
        n_bins = len(adata.obs[stage_label].unique())
    else:
        n_bins = n_bins2
    cut_bins2 = pd.cut(np.array(sub_adata.obs[stage_label]), n_bins, labels=False)

    sub_adata.obs[stage_label] = np.array(sub_adata.obs[stage_label])
    sub_adata.obs['bin']=cut_bins2
    x2 = sub_adata.obs.groupby('bin')[stage_label].mean().values

    for t in np.sort(np.unique(cut_bins2)):
        mean_rep2.append(np.array(sub_adata.obsm[basis][cut_bins2==t].mean(0)).flatten())
    cormat = fast_corrcoef(np.vstack(mean_rep1), np.vstack(mean_rep2))
      
    x1_new = np.linspace(min(x1), max(x1), num)
    x2_new = np.linspace(min(x2), max(x2), num)
    X, Y = np.meshgrid(x1_new, x2_new, indexing='ij')
    interp_func = interpolate.RegularGridInterpolator((x2, x1),cormat, method='linear') # linear, cubic, quintic
    cormat_interp = pd.DataFrame(interp_func((X, Y)), index = np.round(x1_new,2), columns = np.round(x2_new,2))
    if metric == 'similarity':
        ax = sns.heatmap(cormat_interp,cmap = cmap,ax=ax,**kwargs)
    if metric == 'distance':
        ax = sns.heatmap(1-cormat_interp,cmap = cmap,ax=ax,**kwargs)
    ax.set_ylabel(condition1)
    ax.set_xlabel(condition2)
    ax.plot([0, len(x1_new)], [0, len(x2_new)], color='black', linewidth=1, linestyle='--')
    ax.set_title(title)
    if save is not None:
        plt.savefig(save)