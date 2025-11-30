from pandas.io.formats.format import DataFrameFormatter
from sklearn.preprocessing import scale
from fbpca import pca
from scipy.linalg.blas import sgemm
from scipy.sparse import issparse
from collections import OrderedDict
from scipy.sparse import find, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.neighbors import LocalOutlierFactor
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
from .pp import runPCA
mpl.rcParams['pdf.fonttype'] = 42


###############################################################################
def cellSets(adata, 
             celltypes,
             trajectory_label,
             day_field,
             contamination=0.2, 
             run_pca=False):
    """
    Identify inlier cells for each cell type at every time point by removing outliers, 
    to avoid the impact of misclassified cells on trajectory inference.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix containing expression profiles and metadata.
        Must include observation (obs) annotations for cell types and time points.
    celltypes : list of str
        List of target cell type labels to process (corresponding to trajectory_label in adata.obs).
    trajectory_label : str
        Column name in adata.obs that stores cell type annotations for trajectory analysis.
    day_field : str
        Column name in adata.obs that stores time point information (e.g., days/hours of sampling).
    contamination : float, optional (default=0.2)
        Proportion of outliers expected in the dataset (range: 0 to 0.5). 
        Used to tune the Local Outlier Factor (LOF) model for outlier detection.
    run_pca : bool, optional (default=False)
        Whether to perform PCA dimensionality reduction:
        - True: Force PCA computation even if X_pca exists in adata.obsm
        - False: Use existing X_pca if available, otherwise run PCA

    Returns
    -------
    cell_sets : dict
        Dictionary with cell type labels as keys and lists of inlier cell barcodes as values.
        Each entry contains valid (non-outlier) cells for the corresponding cell type across all time points.

    Notes
    -----
    1. Outlier detection is performed using Local Outlier Factor (LOF) on PCA-reduced expression space:
       - LOF measures local deviation of a cell's density compared to its neighbors
       - Neighbor count for LOF is set to min(50% of cells in the subset, 10) to balance sensitivity
    2. PCA is run with 30 components if not precomputed (stored in adata.obsm['X_pca'])
    3. Only inlier cells (pred == 1 from LOF) are retained for subsequent trajectory inference
    """
    cell_sets = {}
    for i in celltypes:
        cell_sets[i] = []
        for t in np.unique(adata.obs[day_field]):
            cells = adata.obs_names.values[(adata.obs[trajectory_label]==i) & (adata.obs[day_field]==t)]
            if len(cells) > 0:
                sub_adata = adata[cells]
                if run_pca | ('X_pca' not in list(sub_adata.obsm.keys())):
                    sub_adata = runPCA(sub_adata,n_components=30)
                X_rep = sub_adata.obsm['X_pca']
                # outliers
                clf = LocalOutlierFactor(n_neighbors=min(int(len(cells)/2),10), contamination = contamination)
                pred = clf.fit_predict(X_rep)
                cell_sets[i] = cell_sets[i] + list(cells[pred==1])
    return(cell_sets)


def trajectory_score(adata, 
                    tmap_model,
                    day_field,
                    trajectory_label,
                    celltypes, 
                    min_cells = 10,
                    contamination=0.2, 
                    run_pca=False, 
                    mutually_exclusion=True, 
                    filter_low_score = True):     
    """
    Calculate trajectory scores for specified cell types across time points using TMAP model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing single-cell expression data, 
        with observation metadata including time and cell type labels.
    tmap_model : TMAP
        Pre-trained TMAP (Trajectory Mapping) model for trajectory inference.
    day_field : str
        Column name in adata.obs representing time points (e.g., days/hours).
    trajectory_label : str
        Column name in adata.obs representing cell type annotations for trajectory analysis.
    celltypes : str or list of str
        Target cell type(s) to analyze trajectory scores for.
    min_cells : int, optional (default=10)
        Minimum number of cells required for a cell type at a specific time point;
        cell groups below this threshold will be filtered out.
    contamination : float, optional (default=0.2)
        Contamination threshold for cell set construction, controlling the maximum
        proportion of non-target cells allowed in each cell set.
    run_pca : bool, optional (default=False)
        Whether to perform PCA dimensionality reduction during cell set construction.
    mutually_exclusion : bool, optional (default=True)
        [Reserved parameter] Whether to enforce mutual exclusion between cell sets (not used in current implementation).
    filter_low_score : bool, optional (default=True)
        [Reserved parameter] Whether to filter out cells with low trajectory scores (not used in current implementation).

    Returns
    -------
    trajectory_ds : AnnData
        AnnData object containing normalized trajectory scores for each cell, with:
        - X: Matrix of trajectory scores (cells × cell types)
        - obs: Metadata with time point information (day_field)
        - var: Cell type labels as variable names

    """
    if type(celltypes) is str:
        celltypes = [celltypes]
    unique_labels = pd.Series(['%s-%s'%(adata.obs[day_field][i], 
                                adata.obs[trajectory_label][i]) for i in range(adata.shape[0])])
    counts = unique_labels.value_counts()
    confuse = counts[counts < min_cells].index.tolist()
    if len(confuse) > 0:
        print(confuse, 'are less than %s cells and removed from adata.'%min_cells)
        adata = adata[~unique_labels.isin(confuse)]

    # self.celltypes = np.array(celltypes)[pd.Series(celltypes).isin(self.adata.obs[self.trajectory_label])]
    # 1. Selecot cell set
    cell_sets = cellSets(adata, 
                            celltypes,
                            trajectory_label,
                            day_field,
                            contamination=contamination,
                            run_pca=run_pca)
    
    # 2. Infer trajectory score
    times = adata.obs[day_field][adata.obs[trajectory_label].isin(celltypes)]
    all_time = np.array(adata.obs[day_field].unique())
    available_time = np.sort(all_time[(all_time >= min(times)) & (all_time <= max(times))])
    scores = {}
    counts = {}
    for t in available_time:
        print('Inference from time: %s'%t)
        sub_populations = tmap_model.population_from_cell_sets(cell_sets, at_time=t)
        sub_trajectory_ds = tmap_model.trajectories(sub_populations)
        sub_trajectory_ds = sub_trajectory_ds[adata.obs_names.tolist()]
        for i in sub_trajectory_ds.var_names:
            if i in list(scores.keys()):
                scores[i] += sub_trajectory_ds[:,i].X.toarray().flatten()
                counts[i] += 1
            else:
                scores[i] = sub_trajectory_ds[:,i].X.toarray().flatten()
                counts[i] = 1

    for i in scores:
        scores[i] = scores[i]/counts[i]
    trajectory_ds = anndata.AnnData(X = np.vstack([scores[i] for i in scores]).T, 
                                        obs = adata.obs.loc[:,[day_field]], 
                                        var = pd.DataFrame(index=list(scores.keys())))
    return(trajectory_ds)
    
def hstack_adata(adata1, adata2):
    traj_name1 = adata1.var_names.tolist()
    traj_name2 = adata2.var_names.tolist()
    basenames = [del_name_number(n) for n in traj_name1]
    basenames_count = pd.Series(basenames).value_counts()
    for i in range(len(traj_name2)):
        if traj_name2[i] in basenames:
            traj_name2[i] = traj_name2[i]+f' ({basenames_count[traj_name2[i]]})'
    combind_name = traj_name1 + traj_name2
    X = np.hstack([adata1.X, adata2.X])
    merge_adata = anndata.AnnData(X = X, 
                                    obs = adata1.obs, 
                                    var = pd.DataFrame(index=combind_name))
    return(merge_adata)


def del_name_number(x):
    return(re.sub(' \([0-9]*\)$', '',x))


def dfs_search(graph, node):
    visited = []
    stack = [node]

    while stack:
        current_node = stack.pop()
        visited.append(current_node)
        neighbors = graph.neighbors(current_node)
        stack.extend(neighbors)

    return visited[1:] 

def path_search(graph, source, target):
    for s in source:
        try:
            visited = nx.shortest_path(graph, source=s, target=target)
        except:
            pass
    return(visited)
