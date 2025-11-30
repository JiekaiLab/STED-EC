# -*- coding: utf-8 -*-
"""
Created on Apr 26 11:48:00 2020

@author: lhlin

Trajectory inference helper function.
"""

import re
import os
import itertools
import numpy as np
import scanpy as sc
import pandas as pd
import nmslib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import concurrent.futures
import h5py
import scipy.sparse as sp
from pandas.api.types import is_categorical_dtype
from scipy.sparse import csr_matrix,find,issparse
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def split(x,f):
    '''
    Split divides the data in the array x into the groups defined by f. 
    Parameters:
    -----
    x: array or DataFrame containing values to be divided into groups.
    f: An array or pandas categorical define the group.
    Returns:
    -----
    d: dict
    A dict of split results.
    '''
    if isinstance(x, pd.DataFrame):
        d = {k: x[f==k] for k in np.unique(f)}
    else:
        d = {k: list(zip(*g))[1] for k, g in itertools.groupby(sorted(zip(f,x)), lambda x: x[0])}
    if is_categorical_dtype(f):
        d = {k:d[k] for k in f.cat.categories.tolist()}
    return(d)

def cmap():
    colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,20))
    colors2 = plt.cm.Reds(np.linspace(0.0, 1, 100))
    colorsComb = np.vstack([colors1, colors2])
    return(LinearSegmentedColormap.from_list('my_colormap', colorsComb))

# Draw a statistical graph
def pl_violin(X=None, Keys=[]):
    i=0
    fig, ax = plt.subplots(nrows=1, ncols=len(Keys), figsize=(9, 4), sharey=False)
    for ax_i in ax:
        ax_i.set_title(Keys[i])
        parts = ax_i.violinplot(X[Keys[i]], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        quartile1, medians, quartile3 = np.percentile(X[Keys[i]], [25, 50, 75])
        ax_i.scatter(1, medians, marker='o', color='white', s=30, zorder=3)
        ax_i.vlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax_i.vlines(1, X[Keys[i]].min(), X[Keys[i]].max(), color='k', linestyle='-', lw=1)
        ax_i.set_xticks([])
        ax_i.grid(axis="y")
        i+=1
    plt.tight_layout()

def knn_outliers(adata,
                 n1=1,
                 n2=20,
                 xmax=18,
                 slope_r=1.8,
                 inter_r=3.9,
                 slope_b=1.0,
                 inter_b=10):
    """
    
    Detect outliers in single cell data using kNN graph in scanpy.uns['neighbors']. The inlier and outlier label is stored in adata.obs['knn_outlier'].
    
    Parameters:
    ----------
    adata : anndata.AnnData.
    n1 : first neighbor index.
    n2 : second neighbor index.
    xmax : maximum of x intercept.
    slope_r : slope of red line.
    inter_r : intercept of red line.
    slope_b : slope of blue line.
    inter_b : intercept of blue line.
    ----------
    
    Usage:
    ------
    >>> import trajectory as ti
    >>> adata=ti.knn_outliers(adata,n1=1,n2=20,xmax=25,slope_r=2.2,inter_r=2.,slope_b=1.2,inter_b=12)
    ------
    """
    
    if "neighbors" not in adata.uns.keys():
        print('adata.uns["neighbors"] is empty, compute neighbors first！！！')
        exit()
    
    #将distances稀疏矩阵提取出来，为CSR格式：
    sparseCsr=adata.uns["neighbors"]["distances"]
    
    dd={} #将行号作为key，行中非零值作为value
    row=0
    for i in range(len(sparseCsr.indptr)-1): # 遍历每一行
        row_star=sparseCsr.indptr[i]
        row_end=sparseCsr.indptr[i+1]
        row_data=sparseCsr.data[row_star:row_end]#取出每行的非零元素
        dd[row]=row_data
        dd[row].sort()
        row=row+1

    #画图：
    x=[] #k=1的距离
    for i in range(len(dd)):
        x.append(dd[i][n1-1]) #将每行元素的最小值放到x中
    x = np.array(x)

    y=[] #k=20的距离
    for i in range(len(dd)):
        y.append(dd[i][n2-1])
    y = np.array(y)
    
    # Calculate the point density
    #xy = np.vstack([x,y])
    #z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6,6))
    #plt.scatter(x, y, c=z, s=10, edgecolor='') 
    plt.hist2d(x,y,bins=100,norm=LogNorm())
    plt.xlim(0, max(x))
    plt.ylim(0, max(y))

    y1=slope_r * x + inter_r
    y2=slope_b * x + inter_b

    plt.plot(x,y1,'r')
    plt.plot(x,y2,'b')
    plt.axvline(x=xmax,ls="-",c="green")

    #设置刻度标记的大小
    plt.tick_params(axis='both',which='major',labelsize=10)


    #设置标题并加上轴标签
    plt.title("Identifying Outliers by k-NN Distance.",fontsize=14)
    plt.xlabel("Distance to neighbor 1",fontsize=14)
    plt.ylabel("Distance to neighbor 20",fontsize=14)


    trim_green  = x < xmax
    trim_red = y < slope_r * x + inter_r
    trim_blue = y < slope_b * x + inter_b

    which = lambda lst:list(np.where(lst)[0])

    cells=list(set(which(trim_green)).intersection(set(which(trim_red))).intersection(set(which(trim_blue))))
    cells = adata.obs.index.values[cells]
    # print("Number of remaining cells:",len(cells))
    plt.show()
    
    # knn_outlier = np.array(['outlier']*adata.shape[0])
    # knn_outlier[adata.obs.index.isin(cells)] = 'inlier'
    n_outlier = adata.shape[0] - len(cells)
    
    adata.obs['knn_inlier'] = adata.obs.index.isin(cells)
    print('kNN distance detects',n_outlier,'outliers ~',np.round(n_outlier/adata.shape[0],4))
    print("Store in .obs['knn_inlier']")

    return adata

def lof_outliers(adata, 
                 n_neighbors:int = 20,
                 contamination:str = 'auto', 
                 n_jobs:int = 10):

    """
    
    Unsupervised Outlier Detection using Local Outlier Factor (LOF). The inlier and outlier label is stored in adata.obs['lof_outlier'].
    
    Parameters:
    ----------
    adata : anndata.AnnData.
    n_neighbors : Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, all samples will be used.
    contamination : The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    n_jobs : The number of parallel jobs to run for neighbors search.
    ----------
    
    Usage:
    ------
    >>> import trajectory as ti
    >>> adata=ti.lof_outliers(adata, n_neighbors=20)
    ------
    """
    
    if "neighbors" not in adata.uns.keys():
        print('adata.uns["neighbors"] is empty, compute neighbors first！！！')
        exit()
    knn = adata.uns["neighbors"]['distances'].indptr[1] - adata.uns["neighbors"]['distances'].indptr[0]
    if (n_neighbors > knn):
        print('n_neighbors is larger than n_neighbors of distance, reset n_neighbor to ', knn)
        n_neighbors = knn
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=n_jobs, metric='precomputed')
    idx = clf.fit_predict(adata.uns['neighbors']['distances'])
    print('LocalOutlierFactor detects',np.sum(idx==-1),'outliers ~',np.round(np.sum(idx==-1)/adata.shape[0],4))
    print("Store in .obs['lof_intlier']")
    adata.obs['lof_intlier'] = np.array(idx) == 1
    
    return adata   

# Find markers
# 把scanpy差异分析结果转成数据框
def extractDEG(adata):
    keys = adata.uns["rank_genes_groups"].keys()
    data = adata.uns["rank_genes_groups"]

    deg_dict = {}
    for key in keys:
        pval = []
        grs = []
        if isinstance(data[key], np.ndarray):
            for k in data[key].dtype.names:
                for i in range(len(data[key][k])):
                    grs.append(k)
                data[key][k].tolist()
                pval.extend(data[key][k].tolist())
            deg_dict[key] = pval
            deg_dict["group"] = grs
    deg=pd.DataFrame(deg_dict)
    return(deg)
    
def nonzero(X):
    # 统计非0表达细胞数
    cs = X.tolil().T.rows
    ns = np.array([len(i) for i in cs])
    return(ns)

def computeCV(X):
    mean = np.array(X.mean(0).tolist()[0])
    sq_mean = mean**2
    X.data = X.data**2
    exp_sq_X = np.array(X.mean(0).tolist()[0])
    std = np.sqrt(exp_sq_X - sq_mean)
    cv = std / mean
    return(mean, cv)
    
def FindAllMarkers(adata, 
                   groupby:str, 
                   use_raw:bool=True,
                   cells_per_group:int=500,
                   seed:int=123,
                   n_genes:int = 200,
                   method:str='wilcoxon',
                   logfc_threshold:float = 0.2,
                   min_pct:float = 0.1,
                   min_diff_pct:float = 0.2,
                   use_rep:str = 'X_pca',
                   n_neighbors:int = 10,
                   sort_values:str = 'diff_pct',
                   deg_id='deg',
                   verbose:bool = False):
    """
    
    Finds markers (differentially expressed genes) for each of the group in a dataset.

    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The key of the observations grouping to consider.
    use_raw : Use `raw` attribute of `adata`.
    cells_per_group : Only perform testing for random smapling cells of each group. If cells_per_group==0, all cells will be used.
    seed : Random seed for sampling.
    n_genes : The number of genes that appear in the returned tables.
    method : Statistic method. One of ['logreg', 't-test', 'wilcoxon', 't-test_overestim_var'].
    logfc_threshold : Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells.
    min_pct : Select genes that are detected at least min_pct fraction in cell type specific group.
    min_diff_pct : Select genes with a minimum difference > min_diff_pct in the fraction of detection between the two groups.
    moran_index : Calculate Moran index. Clustered if I is close to -1, dispersed if I is close to -1, random distributed if I is close to 0.
    use_rep : Low dimension representation for calculating the weights. The weights is 0 if the two cells are not neighbors.
    n_neighbors : n_neighbors for construction of kNN network. 
    sort_values : Sort results by sort_values.
    verbose : verbose
    ----------
    
    Usage:
    ------
    >>> import trajectory as ti
    >>> deg = de.FindAllMarkers(adata, 
                                groupby='leiden', 
                                use_raw=True, 
                                cells_per_group=500,
                                n_genes=200,
                                method='wilcoxon',
                                logfc_threshold=0.25,
                                min_pct=0.1,
                                min_diff_pct=0.3,
                                verbose=False)
    ------
    """
    if cells_per_group > 0:
        if verbose:
            print(f'Sampling {cells_per_group} cells per group')
        cells = adata.obs.index.values
        index = np.array([])
        group_list = adata.obs[groupby].unique()
        for x in group_list:
            pool = cells[adata.obs[groupby] == x]
            n_samples = min(len(pool), cells_per_group)
            np.random.seed(seed)
            index = np.concatenate([index, np.random.choice(pool, n_samples, replace=False)])
        adata1 = adata[index].copy()
    print('Calculate DEGs...')
    if adata1.X.max() > 20:
        use_raw = True
    if adata1.raw is None:
        use_raw = False
    else:
        if adata1.raw.X.max() > 20:
            use_raw = False
    sc.tl.rank_genes_groups(adata1, groupby, use_raw=use_raw, n_genes=n_genes, method=method)
    deg = extractDEG(adata1)
    
    
    if use_raw:
        obj = adata1.raw
    else:
        obj = adata1
    
    pct_1 = np.array([])
    pct_2 = np.array([])
    diff_pct = np.array([])
    group_mean = np.array([])
    group_cv = np.array([])
    for ct in deg['group'][~deg['group'].duplicated()].values:
        if verbose:
            print(f'Compute expr pct: {ct}')
        genes = deg['names'][deg['group']==ct].values
        adata_target = obj[adata1.obs[groupby] == ct, genes]
        adata_other = obj[adata1.obs[groupby] != ct, genes]
        nzero_target_pct = nonzero(adata_target.X)/adata_target.X.shape[0] # Warning: adata.raw.shape[0] != adata.raw.X.shape[0]
        nzero_other_pct = nonzero(adata_other.X)/adata_other.X.shape[0]
        pct_1 = np.concatenate([pct_1,nzero_target_pct])
        pct_2 = np.concatenate([pct_2,nzero_other_pct])
        
        # calculate CV
        sub_X = adata_target.X.copy()
        mean, cv = computeCV(sub_X)
        group_mean = np.concatenate([group_mean,mean])
        group_cv = np.concatenate([group_cv,cv])
                        
    diff_pct = np.concatenate([diff_pct, pct_1-pct_2])
    deg['pct_1'] = pct_1
    deg['pct_2'] = pct_2
    deg['diff_pct'] = diff_pct
    deg['group_mean'] = group_mean
    deg['group_cv'] = group_cv
    
    deg = deg.groupby('group').apply(lambda x: x.sort_values(sort_values, ascending=False))
            
    # Filtering
    deg = deg[deg['logfoldchanges'] > logfc_threshold]
    deg = deg[deg['pct_1'] > min_pct]
    deg = deg[deg['diff_pct'] > min_diff_pct]
    deg.reset_index(inplace=True,drop=True)

    # Set precious
    deg.scores = deg.scores.apply(lambda x: '{:.2f}'.format(x))
    deg.pvals = deg.pvals.apply(lambda x: '{:.3g}'.format(x))
    deg.pvals_adj = deg.pvals_adj.apply(lambda x: '{:.3g}'.format(x))
    deg.logfoldchanges = deg.logfoldchanges.apply(lambda x: '{:.2f}'.format(x))
    deg.pct_1 = deg.pct_1.apply(lambda x: '{:.2f}'.format(x))
    deg.pct_2 = deg.pct_2.apply(lambda x: '{:.2f}'.format(x))
    deg.diff_pct = deg.diff_pct.apply(lambda x: '{:.2f}'.format(x))
    deg.group_mean = deg.group_mean.apply(lambda x: '{:.2f}'.format(x))
    deg.group_cv = deg.group_cv.apply(lambda x: '{:.2f}'.format(x))

    if is_categorical_dtype(adata1.obs[groupby]):
        split_df = split(deg, deg['group'])
        group_order = adata1.obs[groupby].cat.categories
        group_order = group_order[group_order.to_series().isin(deg['group'].unique())]
        deg = pd.concat({i:split_df[i] for i in group_order},ignore_index=True)
    adata.uns[deg_id] = deg
    adata.uns['deg_group'] = groupby
    if verbose:
        print('DEGs is stored in adata.uns["%s"]'%uns_id)
    return(deg)




def get_hp_genes(terms,fuzzy = False):
    '''
    Get gene from Human phenotype database by term or id
    terms: list of term names or ids
    fuzzy: search method for term name. If fuzzy, I will find the term with name contains in given name
    '''
    if isinstance(terms, str):
        terms = [terms]
    df = pd.read_csv('/data1/home/jiazi/lhlin/DATA/database/Phenotype/Human/phenotype_to_genes.csv',index_col = 0)
    id2genes = split(df['gene_symbol'],df['hpo_id'])
    name2genes = split(df['gene_symbol'],df['hpo_name'])
    genes = {}
    for term in terms:
        try:
            if term.startswith("HP:"):
                genes[term] = np.unique(id2genes[term])
            else:
                if fuzzy:
                    fterms = np.array(list(name2genes.keys()))[pd.Series(name2genes.keys()).str.contains(term)]
                    genes[term] = np.unique(np.concatenate([name2genes[i] for i in fterms]))
                else:
                    genes[term] = np.unique(name2genes[term])
        except:
            print('No HP id is found for %s'%term)
    return(genes)

def get_kegg_genes(terms):
    '''
    Get genes from KEGG database by term or id
    terms: list of term name or id
    species: mouse or human
    fuzzy: search method for term name. If fuzzy, I will find the term with name contains in given name
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if isinstance(terms, str):
        terms = [terms]
    robjects.r('options(warn=-1);options(error=NULL)')
    importr('KEGGREST');importr('org.Hs.eg.db')
    robjects.r('hsa_kegg=clusterProfiler::download_KEGG("hsa");PATH2ID=hsa_kegg$KEGGPATHID2EXTID;PATH2NAME=hsa_kegg$KEGGPATHID2NAME',print_r_warnings=False)
    robjects.r('PATH_ID_NAME=merge(PATH2ID, PATH2NAME, by="from");colnames(PATH_ID_NAME)=c("KEGGID", "ENTREZID", "DESCRPTION")',print_r_warnings=False)
    robjects.r('PATH_ID_NAME$GENE=mapIds(org.Hs.eg.db, PATH_ID_NAME$ENTREZID, "SYMBOL", "ENTREZID")',print_r_warnings=False)
    robjects.r('id2gene=split(PATH_ID_NAME$GENE,PATH_ID_NAME$KEGGID);name2gene=split(PATH_ID_NAME$GENE,PATH_ID_NAME$DESCRPTION)',print_r_warnings=False)
    genes = {}
    for term in terms:
        kegg_id = robjects.StrVector([term])
        if term.startswith("hsa"):
            genes[term] = np.unique(list(robjects.r('id2gene[[%s]]'%kegg_id.r_repr())))
        else:
            genes[term] = np.unique(list(robjects.r('name2gene[[%s]]'%kegg_id.r_repr())))
    return(genes)

def get_ra_genes(terms, species='mouse', fuzzy = False):
    '''
    Get genes from Reactome database by term or id
    terms: list of term name or id
    species: mouse or human
    fuzzy: search method for term name. If fuzzy, I will find the term with name contains in given name
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if isinstance(terms, str):
        terms = [terms]
    if species == 'mouse':
        db = 'Mus musculus'
    if species == 'human':
        db = 'Homo sapiens'
    robjects.r('options(warn=-1);options(error=NULL)')
    importr('ReactomeContentService4R')
    genes = {}
    for term in terms:
        if term.startswith("R-"):
            ra_id = robjects.StrVector([term])
        else:
            ra_id = list(robjects.r('searchQuery(query="%s",species="%s",types=c("Pathway"))$results$entries[[1]]$stId'%(term, db)))
            if not fuzzy:
                ra_id = [ra_id[0]]
            ra_id = robjects.StrVector(ra_id)
        if len(ra_id) == 0:
            print('No RA id is found for <%s>'%term)
        else:
            try:
                genes[term] = np.unique(list(robjects.r('unlist(lapply(%s, function(id) {unlist(event2Ids(event.id=id)$geneSymbol)}))'%(ra_id.r_repr()))))
            except:
                genes[term] = np.array([''])
    return(genes)

def get_go_genes(terms, species='mouse', fuzzy = False, return_goid=False):
    '''
    Get genes from GO database by term or id
    terms: list of term name or id
    species: mouse or human
    fuzzy: search method for term name. If fuzzy, I will find the term with name contains in given name
    return_goid: return GO id
    a = robjects.StrVector(['abc', 'def'])
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if isinstance(terms, str):
        terms = [terms]
    if species == 'mouse':
        db = 'org.Mm.eg.db'
    if species == 'human':
        db = 'org.Hs.eg.db'
    robjects.r('options(warn=-1);options(error=NULL)')
    importr('GO.db');importr(db)
    robjects.r('godb=%sGO2ALLEGS'%re.sub('\.db','',db))
    robjects.r('gosym=%sSYMBOL'%re.sub('\.db','',db))
    genes = {}
    for term in terms:
        if term.startswith("GO:"):
            # go_id = robjects.StrVector([term])
            id1 = robjects.StrVector([term]).r_repr()
            robjects.r('go_id=%s'%id1,print_r_warnings=False)
        else:
            if fuzzy:
                robjects.r('go_id=GOID(GOTERM[grep("%s", Term(GOTERM))])'%term,print_r_warnings=False)
                robjects.r('go_id = go_id[go_id %in% keys(godb)]')
                # print('Find %s terms for fuzzy search of %s'(len(list(go_id)),go_id), term,print_r_warnings=False)
            else:       
                robjects.r('go_id = GOID(GOTERM[Term(GOTERM) == "%s"])'%term)
        go_id = robjects.r('go_id')
        if len(go_id) == 0:
            print('No GO id is found for <%s>'%term)
        else:
            try:
                genes[term] = np.unique(robjects.r('unlist(lapply(go_id, function(id) {unlist(mget(get(id, godb),gosym))}))',print_r_warnings=False))
            except:
                genes[term] = np.array([''])
    if return_goid:
        return(genes,list(go_id))
    else:
        return(genes)

def gsePA(lfc, species='mouse'):
    '''
    Perform GSEA for reactome pathway calculated using logfoldchages.

    Parameters:
    ----------
    lfc: pandas series of logfoldchages.
    species: mouse or human

    Retrun:
    ----------   
    DataFrame of PA enrichment

    Usage:
    ----------
    >>> enrich = de.gsePA(lfc)
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if species == 'mouse':
        db='org.Mm.eg.db'
    if species == 'human':
        db='org.Hs.eg.db'
    robjects.r('library(clusterProfiler);library(ReactomePA);library(%s)'%db,print_r_warnings=False)
    robjects.r('genes=%s;lfc=%s'%(robjects.StrVector(lfc.index.tolist()).r_repr(), robjects.FloatVector(lfc.values).r_repr()),print_r_warnings=False)
    robjects.r('gid=bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = "%s")'%db,print_r_warnings=False)
    robjects.r('deg=data.frame(genes=genes,lfc=lfc);deg$gene_id = as.integer(gid$ENTREZID[match(deg$genes,gid$SYMBOL)])',print_r_warnings=False)
    robjects.r('deg = deg[complete.cases(deg),]',print_r_warnings=False)
    robjects.r('geneList = structure(deg$lfc, names = deg$gene_id);geneList = sort(geneList, decreasing = TRUE)',print_r_warnings=False)
    robjects.r('gsea = gsePathway(geneList,organism = "%s",pvalueCutoff=0.1,pAdjustMethod="BH", verbose=FALSE)@result'%species,print_r_warnings=False)
    robjects.r('newGeneNames = sapply(gsea$core_enrichment, function(gs) {unlist(paste0(gid$SYMBOL[match(strsplit(gs, "/")[[1]], gid$ENTREZID)],collapse = "/"))})',print_r_warnings=False)
    robjects.r('gsea$core_enrichment = newGeneNames',print_r_warnings=False)
    gsea = robjects.r('gsea[order(gsea$NES, decreasing = TRUE), ]',print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
      gsea = robjects.conversion.get_conversion().rpy2py(gsea)
    gsea['Database'] = 'PA'
    if 'core_enrichment' in gsea.columns.tolist():
        gsea.rename(columns = {'core_enrichment':'geneID'},inplace = True)
        gsea['Count'] = [len(x.split('/')) for x in gsea['geneID']]
    return(gsea)

def gseGO(lfc, species='mouse'):
    '''
    Perform gene set enrichment for GO terms calculated using logfoldchages.

    Parameters:
    ----------
    lfc: pandas series of logfoldchages.
    species: mouse or human

    Retrun:
    ----------   
    DataFrame of GO enrichment

    Usage:
    ----------
    >>> enrich = de.gseGO(lfc)
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if species == 'mouse':
        db='org.Mm.eg.db'
    if species == 'human':
        db='org.Hs.eg.db'
    robjects.r('library(clusterProfiler);library(%s)'%db,print_r_warnings=False)
    robjects.r('genes=%s;lfc=%s'%(robjects.StrVector(lfc.index.tolist()).r_repr(), robjects.FloatVector(lfc.values).r_repr()),print_r_warnings=False)
    robjects.r('gid=bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = "%s")'%db,print_r_warnings=False)
    robjects.r('deg=data.frame(genes=genes,lfc=lfc);deg$gene_id = as.integer(gid$ENTREZID[match(deg$genes,gid$SYMBOL)])',print_r_warnings=False)
    robjects.r('deg = deg[complete.cases(deg),]',print_r_warnings=False)
    robjects.r('geneList = structure(deg$lfc, names = deg$gene_id);geneList = sort(geneList, decreasing = TRUE)',print_r_warnings=False)
    robjects.r('gsea = gseGO(geneList, ont="BP",OrgDb = "%s",pvalueCutoff=0.1,pAdjustMethod="BH", verbose=FALSE)@result'%db,print_r_warnings=False)
    robjects.r('newGeneNames = sapply(gsea$core_enrichment, function(gs) {unlist(paste0(gid$SYMBOL[match(strsplit(gs, "/")[[1]], gid$ENTREZID)],collapse = "/"))})',print_r_warnings=False)
    robjects.r('gsea$core_enrichment = newGeneNames',print_r_warnings=False)
    gsea = robjects.r('gsea[order(gsea$NES, decreasing = TRUE), ]',print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
      gsea = robjects.conversion.get_conversion().rpy2py(gsea)
    gsea['Database'] = 'GO'
    if 'core_enrichment' in gsea.columns.tolist():
        gsea.rename(columns = {'core_enrichment':'geneID'},inplace = True)
        gsea['Count'] = [len(x.split('/')) for x in gsea['geneID']]
    return(gsea)

def enrichGO(deg,species='mouse'):
    '''
    Perform GO enrichment for deg calculated by de.FindAllMarkers using clusterProfiler in R.

    Parameters:
    ----------
    deg: pandas DataFrame calculated by de.FindAllMarkers
    species: mouse or human

    Retrun:
    ----------   
    DataFrame of GO enrichment

    Usage:
    ----------
    >>> deg = de.FindAllMarkers(adata, group = 'leiden')
    >>> enrich = de.enrichGO(adata)
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if species == 'mouse':
        db='org.Mm.eg.db'
    if species == 'human':
        db='org.Hs.eg.db'
    robjects.r('names=%s;group=%s'%(robjects.StrVector(deg['names'].tolist()).r_repr(),robjects.StrVector(deg['group'].tolist()).r_repr()))
    robjects.r('library(clusterProfiler);library(ReactomePA);library(%s)'%db,print_r_warnings=False)
    robjects.r('gcSample=lapply(split(names, group), function(gr) as.numeric(bitr(gr,fromType="SYMBOL", toType="ENTREZID", OrgDb=%s)$ENTREZID))'%db,print_r_warnings=False)
    go_res=robjects.r('compareCluster(gcSample, OrgDb=%s, fun="enrichGO", pvalueCutoff=0.1, qvalueCutoff=0.1, ont="BP", readable=T)@compareClusterResult'%db,print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
      go_res = robjects.conversion.get_conversion().rpy2py(go_res)
    go_res['Database'] = 'GO'
    go_res['GeneRatio'] = [int(i.split('/')[0])/int(i.split('/')[1]) for i in go_res['GeneRatio']]
    return(go_res)

def enrichPA(deg,species='mouse'):
    '''
    Perform Reactome pathway enrichment for deg calculated by de.FindAllMarkers using clusterProfiler in R.

    Parameters:
    ----------
    deg: pandas DataFrame calculated by de.FindAllMarkers
    species: mouse or human

    Retrun:
    ----------   
    DataFrame of Reactome pathway enrichment

    Usage:
    ----------
    >>> deg = de.FindAllMarkers(adata, group = 'leiden')
    >>> enrich = de.enrichPA(adata)
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if species == 'mouse':
        db='org.Mm.eg.db'
    if species == 'human':
        db='org.Hs.eg.db'
    robjects.r('names=%s;group=%s'%(robjects.StrVector(deg['names'].tolist()).r_repr(),robjects.StrVector(deg['group'].tolist()).r_repr()))
    robjects.r('library(clusterProfiler);library(ReactomePA);library(%s)'%db,print_r_warnings=False)
    robjects.r('gcSample=lapply(split(names, group), function(gr) as.numeric(bitr(gr,fromType="SYMBOL", toType="ENTREZID", OrgDb=%s)$ENTREZID))'%db,print_r_warnings=False)
    pa_res=robjects.r('compareCluster(gcSample,organism="%s",fun="enrichPathway",pvalueCutoff=0.1,qvalueCutoff=0.1,readable=T)@compareClusterResult'%species,print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
      pa_res = robjects.conversion.get_conversion().rpy2py(pa_res)
    pa_res['Database'] = 'PA'
    pa_res['GeneRatio'] = [int(i.split('/')[0])/int(i.split('/')[1]) for i in pa_res['GeneRatio']]
    return(pa_res)


def cal_lfc(adata, group, use_hvg = False, lfc_id='lfc'):
    '''
    Calculate logfoldchage for each group of cells of single cell data.
    Parameters:
    ----------
    adata: anndata
    group: Compararison of each group in adata.obs[group].
    use_hvg: Only calculate the 
    highly variable genes.
    lfc_id: Key of adata.uns to store results.
    
    Retrun:
    ----------   
    Dict of logfoldchage of each group.

    Usage:
    ----------
    >>> lfc = de.cal_lfc(adata, 'leiden')
    '''
    groups = np.unique(adata.obs[group])
    if use_hvg:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sub_adata = adata[:,adata.var['highly_variable']]
    else:
        sub_adata = adata
    genes = sub_adata.var_names.tolist()
    lfc = {}
    for x in groups:
        g1 = np.array(sub_adata.X[sub_adata.obs[group] == x].mean(0)).flatten()
        g2 = np.array(sub_adata.X[sub_adata.obs[group] != x].mean(0)).flatten()
        lfc[x] = pd.Series(g1-g2,index = genes)
    adata.uns[lfc_id] = lfc
    return(lfc)

def EnrichPathway(adata,
                  deg=None,
                  groupby = None,
                  use_hvg = False,
                  species='auto',
                  method='clusterprofiler',
                  algorithm='fish',
                  deg_id='deg',
                  enrich_id='enrich',
                  filename=None,
                  verbose = False):
    '''
    Pathway enrichment for degs found by de.FindAllMarkers.
    
    Parameters:
    ----------
    adata: anndata.AnnData.
    deg: pandas dataframe of degs, if None, I will use deg result in adata.uns[deg_id].
    groupby: group in adata.obs. Used for calculating logfoldchages when algorithm is gsea.
    species: Species name for database selection for enrichment. Can be 'auto', 'mouse' and 'human'.
    method: gprofiler or clusterprofiler. gprofiler is much faster and more informative, while clusterprofiler is more precies.
    algorithm: fish or gsea. FISH testing or Gene Set Enrichment Analysis.
    deg_id: name of deg analysis result in adata.uns, only used when deg is None.
    enrich_id: name of enrichment analysis result in adata.uns, default is 'enrich'.
    filename: If not None, export enrichment result as HTML.
    verbose: verbose
    
    Return:
    ----------
    DataFrame of enrichment result, including GO, Reactome, KEGG, Human phenotype, Upstream TF and etc.

    Usage:
    ----------
    >>> deg = de.FindAllMarkers(adata, group = 'leiden')
    >>> enrich = de.EnrichPathway(adata)
    
    Note:
    ----------
    The target genes of GO or Reactome are convinced when using the right species name. However, sc.queries.enrich 
    will enrich huamn database for mouse genes, like KEGG, HP, so the target genes may not incomplete for these database.
    '''
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    if species == 'auto':
        if len(adata.var.index[adata.var.index.str.startswith('MT-')]):
            org = 'hsapiens'
            org_name = 'human'
        elif len(adata.var.index[adata.var.index.str.startswith('mt-')]):
            org = 'mmusculus'
            org_name = 'mouse'
        else:
            org = 'mmusculus'
            org_name = 'mouse'
            print('I dont known what species should be used, I used mouse by default.')
    if species == 'human':
        org = 'hsapiens'
        org_name = 'human'
    if species == 'mouse':
        org = 'mmusculus'
        org_name = 'mouse'
    if deg is None:
        deg = adata.uns[deg_id]
    if algorithm == 'fish':
        if method == 'clusterprofiler':
            print('Enrich for GO terms...')
            res_go = enrichGO(deg, species = org_name)
            print('Enrich for Reactome pathway...')
            res_ra = enrichPA(deg, species = org_name)
            enrich = pd.concat([res_go,res_ra]).reset_index()
            enrich['Method'] = method
            enrich = enrich.loc[:, ['Cluster','Database','ID','Description','GeneRatio','pvalue','p.adjust','qvalue','geneID','Count','Method']]
            enrich.pvalue = enrich.pvalue.apply(lambda x: '{:.2g}'.format(x))
            enrich['p.adjust'] = enrich['p.adjust'].apply(lambda x: '{:.2g}'.format(x))
            enrich.qvalue = enrich.qvalue.apply(lambda x: '{:.2g}'.format(x))
        if method == 'gprofiler':
            markers = split(deg.names,deg.group)
            print('Enrich pathway...')
            enrich = sc.queries.enrich(markers,org = org)
            # Modify dataframe
            enrich['effective_domain_size'] = enrich['intersection_size']/enrich['query_size']
            enrich.rename(columns = {'effective_domain_size':'GeneRatio'},inplace=True)
            enrich['native'] = [re.sub('REAC:','',i) for i in enrich['native']]
            enrich['native'] = [re.sub('KEGG:','hsa',i) for i in enrich['native']]
            enrich['source'] = [re.sub('REAC:','PA',i) for i in enrich['source']]
            # Sort result
            def custom_sort(group):
                return group.sort_values(by='p_value',ascending=True)
            enrich = enrich.groupby('query', group_keys=False).apply(custom_sort)
            # Find genes of GO
            go_terms = enrich['native'][enrich['native'].str.startswith('GO:')].unique()
            if len(go_terms) > 0:
                go_genes = get_go_genes(go_terms,species=org_name)
            else:
                go_genes = {}
            # Find genes of Reactome
            ra_terms = enrich['native'][enrich['native'].str.startswith('R-')].unique()
            if len(ra_terms) > 0:
                ra_genes = get_ra_genes(ra_terms,species=org_name)
            else:
                ra_genes = {}
            # Find genes of KEGG
            kegg_terms = enrich['native'][enrich['native'].str.startswith('hsa')].unique()
            if len(kegg_terms) > 0:
                kegg_genes = get_kegg_genes(kegg_terms)
            else:
                kegg_genes = {}
            # Find genes of Human Phenotype
            hp_terms = enrich['native'][enrich['native'].str.startswith('HP:')].unique()
            if len(hp_terms) > 0:
                hp_genes = get_hp_genes(hp_terms)
            else:
                hp_terms = {}
            # Merge all geneset
            gene_set = go_genes|ra_genes|kegg_genes|hp_genes
            # Filter intersection between degs and term gene sets    
            deg_list = split(deg['names'],deg['group'])
            gnames = []
            for i in range(enrich.shape[0]):
                id = enrich['native'].values[i]
                group = enrich['query'].values[i]
                try:
                    mask = pd.Series([i.upper() for i in deg_list[group]]).isin([i.upper() for i in gene_set[id]])
                    gnames.append('/'.join(np.array(deg_list[group])[mask]))
                except:
                    gnames.append('')
            enrich['geneID'] = gnames
            enrich.rename(columns = {'native':'ID','name':'Description','p_value':'pvalue','effective_domain_size':'GeneRatio','intersection_size':'Count','source':'Database','query':'Cluster'},inplace=True)
            enrich['Method'] = method
            enrich.pvalue = enrich.pvalue.apply(lambda x: '{:.2g}'.format(x))
            enrich = enrich.loc[:, ['Cluster','Database','ID','Description','GeneRatio','pvalue','geneID','Count','Method']]
        enrich.GeneRatio = round(enrich.GeneRatio,2)
    if algorithm == 'gsea':
        if groupby is None:
            groupby=adata.uns['deg_group']
        print('Calculate logfoldchage for %s...'%groupby)
        lfc = cal_lfc(adata, groupby,use_hvg=use_hvg)
        print('Perform GSEA...')
        enrich = []
        for x in lfc:
            if verbose:
                print(x)
            pa = gsePA(lfc[x], species = org_name);pa['Cluster'] = x;pa['Method'] = 'clusterprofiler'
            go = gseGO(lfc[x], species = org_name);go['Cluster'] = x;go['Method'] = 'clusterprofiler'
            enrich.append(pa)
            enrich.append(go)
        enrich = pd.concat(enrich)
        def custom_sort(group):
            return group.sort_values(by='NES',ascending=False)
        enrich = enrich.groupby('Database', group_keys=False).apply(custom_sort)
        enrich.reset_index(inplace = True)
        enrich = enrich.loc[:, ['Cluster','Database','ID','Description','NES','pvalue','geneID','Count','Method']]
    enrich['Description'] = enrich['Description'].apply(lambda x: x.capitalize())
    adata.uns[enrich_id] = enrich
    if verbose:
        print('Enrichment result is stored in adata.uns["%s_%s"]'%(enrich_id, algorithm))
    if filename is not None:
        Cluster=robjects.StrVector(enrich['Cluster']) 
        ID=robjects.StrVector(enrich['ID'])
        Description=robjects.StrVector(enrich['Description'])
        pvalue=robjects.FloatVector(enrich['pvalue'])
        geneID=robjects.StrVector(enrich['geneID'])
        Count=robjects.IntVector(enrich['Count'])
        Database=robjects.StrVector(enrich['Database'])
        Method=robjects.StrVector(enrich['Method'])
        if algorithm == 'fish':
            GeneRatio=robjects.FloatVector(enrich['GeneRatio'])
            robjects.r('df=data.frame(Cluster=as.factor(%s),Database=as.factor(%s),ID=%s,Description=%s,GeneRatio=%s,pvalue=%s,geneID=%s,Count=%s,Method=%s)'%(Cluster.r_repr(),Database.r_repr(),ID.r_repr(),Description.r_repr(),GeneRatio.r_repr(),pvalue.r_repr(),geneID.r_repr(),Count.r_repr(),Method.r_repr()))
        if algorithm == 'gsea':
            NES=robjects.FloatVector(enrich['NES'])
            robjects.r('df=data.frame(Cluster=as.factor(%s),Database=as.factor(%s),ID=%s,Description=%s,NES=%s,pvalue=%s,geneID=%s,Count=%s,Method=%s)'%(Cluster.r_repr(),Database.r_repr(),ID.r_repr(),Description.r_repr(),NES.r_repr(),pvalue.r_repr(),geneID.r_repr(),Count.r_repr(),Method.r_repr()))
        robjects.r('library(DT);datatable(df,filter="top",options=list(pageLength=20))%>%saveWidget("enrich_tmp.html")',print_r_warnings=False)
        # save results
        if os.path.dirname(filename) == '':
            filename = os.path.join('./',filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        os.rename('enrich_tmp.html', filename)
        print('Export result to %s'%filename)
    return(enrich)






