import re,os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_categorical_dtype
from scipy.sparse import csr_matrix,find,issparse
from mpl_toolkits.axes_grid1 import AxesGrid
from seaborn import color_palette
from sklearn.cluster import KMeans
from matplotlib import colors
from matplotlib.font_manager import FontProperties

import matplotlib as mpl
import matplotlib.colors as mcolors
import toytree
import itertools
import scanpy as sc
from toytree.TreeStyle import TreeStyle
import networkx as nx
import pandas as pd
import seaborn as sns
from pygam import GAM, s
import anndata
import warnings
import concurrent.futures
from scipy.linalg import get_blas_funcs
from utils.pp import *
from utils import core

warnings.simplefilter(action='ignore', category=FutureWarning)
def Qi(v1, v2):
    return((3/4)*v1 + (1/4)*v2)
def Ri(v1, v2):
    return((1/4)*v1 + (3/4)*v2)
def smooth_line(verts, niter=5):
    '''
    曲线平滑函数
    '''
    for n in range(niter):
        new_verts = []
        for i in range(len(verts)-1):
            if i == 0:
                new_verts.append(verts[i])
                new_verts.append(Ri(verts[i], verts[i+1]))
            elif i == len(verts)-2:
                new_verts.append(Qi(verts[i], verts[i+1]))
            else:
                new_verts.append(Qi(verts[i], verts[i+1]))
                new_verts.append(Ri(verts[i], verts[i+1]))
        new_verts.append(verts[len(verts)-1])
        verts = np.vstack(new_verts)
    return(verts)

def findHalf(x):
    if x.sum() == 0:
        return(0,len(x)-1)
    else:
        m = (x.max()-x.min())/2
        half = np.where(x > m)[0]
        return((half[0], half[-1]))


def addPath(verts, wdith = 0.3):
    '''
    添加路径
    '''
    x = verts[:,0]
    y = verts[:,1]
    nx = wdith
    ny = wdith
    xp = x.copy() #+ nx
    yp = y.copy() + ny
    xn = x.copy() #- nx
    yn = y.copy() - ny

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    vertices = np.vstack([vertices,vertices[0]])
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(vertices)-1] = Path.MOVETO
    return(Path(vertices, codes))
def recursive_search(dict, key):
    if key in dict:
        return dict[key]
    for k, v in dict.items():
        item = recursive_search(v, key)
        if item is not None:
            return item
def bfs_edge_lst(graph, root):
    return list(nx.bfs_edges(graph, root))
def load_graph(filename):
    G = nx.Graph()
    # build the graph
    return G
def tree_from_edge_lst(elst, root):
    tree = {root: {}}
    for src, dst in elst:
        subt = recursive_search(tree, src)
        subt[dst] = {}
    return tree
def tree_to_newick(tree):
    items = []
    for k in tree.keys():
        s = ''
        if len(tree[k].keys()) > 0:
            subt = tree_to_newick(tree[k])
            if subt != '':
                s += '(' + subt + ')'
        s += k
        items.append(s)
    return ','.join(items)
def nx2newick(G, root):
    elst = bfs_edge_lst(G, root)
    tree = tree_from_edge_lst(elst, root)
    return(tree_to_newick(tree) +';')

def newick2nx(tree):
    tips = tree.get_tip_labels()
    G = nx.DiGraph()
    for n in tips:
        nodes = get_ancestors(tree, n)
        for i in range(len(nodes)-1):   
            G.add_edge(nodes[i],nodes[i+1],char = '')
    return(G)

def check_connection(df):
    if not all(x == df.label[0] for x in df.label):
        add_verts = []
        for i in range(df.shape[0]-1):
            if df.label[i] != df.label[i+1]:
                vert_i0 = df.iloc[i,:]
                vert_i1 = df.iloc[i+1,:]
                add_verts.append(pd.DataFrame({'x':vert_i0.x, 
                                               'y':vert_i0.y, 
                                               'time':vert_i1.time,
                                               'label':vert_i1.label},
                                                index = [df.index.values[i+1]]))
        add_verts.append(df)
        return(pd.concat(add_verts).sort_values('time'))
    else:
        return(df)
    
def get_ancestors(tree, node, ignore_1st=False):
    idx = tree.get_mrca_idx_from_tip_labels(names=[node])
    ancestors = tree.idx_dict[idx].get_ancestors()
    if ignore_1st:
        path = [node]+[ancestors[i].name for i in range(len(ancestors))][:-1]
    else:
        path = [node]+[ancestors[i].name for i in range(len(ancestors))]
    return(path[::-1])

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
    >>> deg = ti.FindAllMarkers(adata, 
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
        adata = adata[index].copy()
    if adata.X.max() > 20:
        use_raw = True
    if adata.raw is None:
        use_raw = False
    else:
        if adata.raw.X.max() > 20:
            use_raw = False
    sc.tl.rank_genes_groups(adata, groupby, use_raw=use_raw, n_genes=n_genes, method=method)
    deg = extractDEG(adata)
    
    
    if use_raw:
        obj = adata.raw
    else:
        obj = adata
    
    pct_1 = np.array([])
    pct_2 = np.array([])
    diff_pct = np.array([])
    group_mean = np.array([])
    group_cv = np.array([])
    for ct in deg['group'][~deg['group'].duplicated()].values:
        if verbose:
            print(f'Compute expr pct: {ct}')
        genes = deg['names'][deg['group']==ct].values
        adata_target = obj[adata.obs[groupby] == ct, genes]
        adata_other = obj[adata.obs[groupby] != ct, genes]
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

    if is_categorical_dtype(adata.obs[groupby]):
        split_df = split(deg, deg['group'])
        group_order = adata.obs[groupby].cat.categories
        group_order = group_order[group_order.to_series().isin(deg['group'].unique())]
        deg = pd.concat({i:split_df[i] for i in group_order},ignore_index=True)

    return(deg)


def batchSmooth(Y, x=None,
                x_pred=None, 
                new_samples=100, 
                n_splines=None,
                spline_order=2,
                max_workers=None):
    
    if issparse(Y):
        Y = Y.A
    if x is None:
        x = np.arange(Y.shape[0])
    if x_pred is None:
        x_pred = np.linspace(min(x), max(x), new_samples)
    if n_splines is None:
        n_splines = min(len(x), 10)
    else:
        n_splines = min(len(x), n_splines)
    params = [(x, x_pred, Y[:,i], n_splines, spline_order) for i in range(Y.shape[1])]
        
    # parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor: 
        results = executor.map(doBatchSmooth, params)
        
    results = [x for x in results]
    return(np.vstack(results).T)

def doBatchSmooth(p):
    gam = GAM(s(feature = 0, 
                penalties= 'derivative', 
                n_splines=p[3], 
                spline_order = p[4]),
                distribution='normal',
                verbose = False).fit(np.atleast_2d(p[0]).T, p[2])        
    y_pred = gam.predict(p[1]) 
    return(y_pred)

def plotDynamicHeatmap(adata_dict, 
                       conditions = None,
                       genes = None, 
                       n_clusters = 3,
                       retun_table = True,
                       label_genes = None,
                       sort_gene_by_trend=True,
                       labels = None,
                       separete_norm = False,
                       cmap_heatmap='YlGnBu_r',
                       cmap_time = 'Spectral_r',
                       figsize = (3,4),
                       axes_pad=0.02,
                       aspect = 'auto',
                       title_font_size = 10,
                       gene_font_size = 8):

    '''
    必须保证所有条件使用相同的基因
    比较不同条件下基因随时间变化的趋势
    '''
    if type(adata_dict) != dict:
        adata_dict = {'':adata_dict}
    if conditions is None:
        conditions = list(adata_dict.keys())
    data = adata_dict[conditions[0]]
    # 0-1标准化数值
    mat_list = []
    first_half = []
    last_half =  []
    mat_mean = []
    for condition in conditions:
        if genes is not None:
            used_genes = np.array(genes)[pd.Series(genes).isin(data.var_names.values)]
            mat = np.array(adata_dict[condition][:, used_genes].X)
        else:
            used_genes = data.var_names.values
            mat = np.array(adata_dict[condition].X)
        # mat_list.append(mat)
        if separete_norm:
            ms = np.apply_along_axis(lambda x: (x-x.min())/(x.max()-x.min()), axis = 0, arr = mat)
            ms[np.isnan(ms)] = 0

            mat_list.append(ms)
        else:
            mat_list.append(mat)
        # 基因的激活时间
        half_inds = np.apply_along_axis(findHalf, axis = 0, arr = mat)
        first_half.append(half_inds[0,:])
        last_half.append(half_inds[1,:])
        mat_mean.append(mat.mean(0))

    # 对合并多种条件的基因表达聚类
    mat = np.concatenate(mat_list, axis = 0)
    if separete_norm:
        mat = mat.T
    else:
        mat = np.apply_along_axis(lambda x: (x-x.min())/(x.max()-x.min()), axis = 0, arr = mat).T
    # --- kmeans
    
    kmeans = KMeans(n_clusters = n_clusters,n_init=50, max_iter=500, random_state = 0).fit(mat) 
    
    # 根据表达值较高的条件对基因排序
    idx_max = pd.DataFrame(np.vstack(mat_mean)).idxmax(axis  = 0)
    first_half = np.vstack(first_half); last_half = np.vstack(last_half)
    first_half = [first_half[idx_max[i],i] for i in range(len(idx_max))]
    last_half = [last_half[idx_max[i],i] for i in range(len(idx_max))]
    
    df = pd.DataFrame({'gene':used_genes, 
                       'first_half':first_half, 
                       'last_half':last_half,
                       'label':kmeans.labels_})

    # --- Sort the gene by first and last half of gene expression
    if sort_gene_by_trend:
        sort_df = df.sort_values(['first_half','last_half'],ascending=True)
    else:
        sort_df = df
    sort_df['ids'] = np.arange(sort_df.shape[0])
    group_order = sort_df.groupby('label')['ids'].mean().sort_values().index.values

    # --- Rename group id to match the gene order
    new_group_labs = dict(zip(group_order,  np.arange(len(group_order))))
    
    sort_df['label'] = pd.Categorical([new_group_labs[i] for i in sort_df['label']],categories = np.arange(len(group_order)))
    sort_df.reset_index(inplace=True,drop=True)



    # --- Aspect
    if aspect == 'auto':
        aspect = len(conditions)


    # --- Get the expression matrix for (clusters[rows], lineage[cols])
    
    # 分割标准化的数据
    parts = np.repeat(conditions,data.shape[0])
    vals = []
    for i in sort_df['label'].cat.categories: # Gene cluster
        for j in conditions: # Lineage
            
            sub_mat = pd.DataFrame(mat[:, parts == j], index = used_genes)
            gs = sort_df['gene'].values[sort_df['label'] == i]
            if (len(conditions) == 2) & (j == conditions[0]):
                vals.append(sub_mat.loc[gs,::-1].values)
            else:
                vals.append(sub_mat.loc[gs,:].values)

    for i in range(len(conditions)):
        levels = np.arange(data.shape[0]).reshape(-1,1)
        if (len(conditions) == 2) & (i == 0):
            vals.append(np.tile(levels, np.ceil(sort_df.shape[0]/50).astype(int)).T[:,::-1])
        else:
            vals.append(np.tile(levels, np.ceil(sort_df.shape[0]/50).astype(int)).T)
        
    
        
        
    # --- Grid heatmap
    fig = plt.figure(figsize=figsize)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(sort_df['label'].cat.categories)+1,
                                 len(conditions)),
                    axes_pad=axes_pad,
                    share_all=False,
                    label_mode="L",
                    aspect=True,
                    cbar_size = '1%',
                    cbar_location='right',
                    cbar_mode="single"
                    )

    seq = 0
    lng = 0
    gene_order = []
    for val, ax in zip(vals,grid):
        if seq > len(conditions)*len(sort_df['label'].cat.categories)-1:
            if type(cmap_time) == str:
                c = create_gradient_color(palette = cmap_time,n=data.shape[0])
            else:
                c = cmap_time
            cmap = mpl.colors.ListedColormap(c)
            barprops = dict(aspect=aspect, cmap=cmap, interpolation='nearest')
            im = ax.imshow(val,**barprops)
            im.set_rasterized(True)
            ax.set_yticklabels([])
        else:
            im = ax.imshow(val, vmin=0, vmax=1,aspect=aspect, cmap=cmap_heatmap)
            im.set_rasterized(True)
            # --- Color bar
            if seq == 0:
                grid.cbar_axes[0].colorbar(im)
            # --- Heatmap frame off
            for spine in ax.spines.values():
                # spine.set_edgecolor('white')
                spine.set_lw(0)
            # --- Heatmap title
            if seq < len(conditions):
                ax.set_title(conditions[seq], size = title_font_size,rotation=30,ha='center')
            # --- Plot genes
            if seq%len(conditions) == 0:
                show_genes = sort_df['gene'].values[sort_df['label'] == sort_df['label'].cat.categories.values[lng]]
                gene_order.append(show_genes)
                if label_genes is not None:
                    show_genes = show_genes.copy()
                    show_genes[~pd.Series(show_genes).isin(np.atleast_1d(label_genes))] = ''
                ax.set_yticks(np.arange(val.shape[0]))
                ax.set_yticklabels(show_genes, size = gene_font_size)
                lng += 1
        ax.grid(None)
        ax.set_xticks([])
        ax.tick_params(axis=u'both', which=u'both', length = 0, pad = 1) # pad: tick到label的距离
        seq += 1
    # --- Color bar
    for cax in grid.cbar_axes:
        cax.tick_params(axis=u'both', which=u'both',length=2,pad = 1)
        cb_yticklabels = np.round(np.linspace(0, 1, len(cax.get_yticklabels())),1)
        cax.set_yticklabels(cb_yticklabels,size = 8)
        cax.toggle_label(True)

    # --- Remove grid
    plt.grid(visible=None)
    # plt.tight_layout()
    
    # --- Returen table that with genes order matched the heatmap
    if retun_table:
        sort_df.set_index('gene', drop=False, inplace=True)
        sort_df = sort_df.loc[np.concatenate(gene_order),:]
        sort_df.reset_index(inplace=True,drop=True)
        sort_df.drop(columns=['ids'], inplace = True)
        
        return(sort_df)

# 加速OT
def moscot_transport(adata, 
                     day_field, 
                     features = None,
                     n_pcs = 40,
                     n_ccs = 20,
                     method = 'cca',
                     tag = 'ott',
                     epsilon=1e-3,
                     tau_a=0.99, 
                     tau_b=0.999, 
                     scale_cost="mean",
                     min_iterations=1, 
                     max_iterations=20):
    import moscot,os
    from moscot.problems.time import TemporalProblem
    os.makedirs('tmaps',exist_ok=True)
    alltimes=np.sort(adata.obs[day_field].unique())
    for i in range(len(alltimes)-1):
        t1 = alltimes[i]
        t2 = alltimes[i+1]
        print('coupling %s-%s'%(t1,t2))
        source = adata.obs_names[adata.obs[day_field] == alltimes[i]]
        target = adata.obs_names[adata.obs[day_field] == alltimes[i+1]]
        if method == 'cca':
            couples = runCCA(adata,
                            source, 
                            target,
                            features = features,
                            standarize = True,
                            n_components = n_ccs, use_pcs=n_pcs)
        elif method == 'pca':
            couples = runPCA(adata,
                            source, 
                            target,
                            features = features,
                            n_components = n_pcs)
        day_field_values = couples.obs[day_field].values
        try:
            couples.obs[day_field] = pd.Categorical(day_field_values, np.sort(np.unique(day_field_values)))
            tp = TemporalProblem(couples).prepare(time_key=day_field, 
                                                joint_attr="X_%s"%method).solve(epsilon=epsilon, 
                                                tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost,
                                                min_iterations=min_iterations, max_iterations=max_iterations)
        except:
            couples.obs[day_field] = np.array(day_field_values)
            tp = TemporalProblem(couples).prepare(time_key=day_field, 
                                                joint_attr="X_%s"%method).solve(epsilon=epsilon, 
                                                tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost,
                                                min_iterations=min_iterations, max_iterations=max_iterations)
        cps_adata = anndata.AnnData(X = csr_matrix(tp.solutions[(t1, t2)].transport_matrix),
                        obs = pd.DataFrame(index=source),var = pd.DataFrame(index=target))
        cps_adata.write('tmaps/%s_%s_%s.h5ad'%(tag, t1,t2))




# 类
class timeSeriesTree:
    def __init__(self, adata = None, tmap_model = None, day_field = None, trajectory_label = None, lineages = None):
        '''
        adata: AnnData
        tmap_model: WOT model
        '''
        if adata is not None:
            self.adata = adata
        if tmap_model is not None:
            self.tmap_model = tmap_model
        if day_field is not None:
            self.day_field = day_field
            if adata is not None:
                self.adata.obs[self.day_field] = pd.Categorical(self.adata.obs[self.day_field])
                self.times = self.adata.obs[self.day_field].cat.categories.sort_values(ascending=False)
        if trajectory_label is not None:
            self.trajectory_label = trajectory_label
        if lineages is not None:
            self.lineages = lineages.copy()
            if adata is None:
                self.buildTree()
        else:
            self.lineages = None
        
    def add_node_type(self,G,name):
        df = self.pos.copy()
        if self.ignore_1st:
            df = df[df.label != 'pseudo_root'].copy()
        self.root = df.index[df.x == df.x.min()].tolist()
        self.leaf = [n for n in dict(G.out_degree()) if dict(G.out_degree())[n] == 0]
        self.branch = [n for n in dict(G.degree()) if dict(G.degree())[n] > 2]
        self.branch = [n for n in self.branch if n not in self.root+self.leaf+['-999.0_pseudo_root']]
        self.joint = [node for node in df.index if (df.loc[node,'label'] not in df.loc[[n for n in G.successors(node)],'label'].tolist()) & (G.out_degree(node) == 1)]
        self.joint = [n for n in self.joint if n not in self.root+self.leaf+self.branch+['-999.0_pseudo_root']]
        self.pos[name] = 'node'
        self.pos.loc[self.pos.index.isin(self.root),name] = 'root'
        self.pos.loc[self.pos.index.isin(self.leaf),name] = 'leaf'
        self.pos.loc[self.pos.index.isin(self.branch),name] = 'branch'
        self.pos.loc[self.pos.index.isin(self.joint),name] = 'joint'

    def set_lineages(self, lineages_path, interpolation = True):
        self.lineages_fixed = pd.read_csv(lineages_path,index_col = 0)
        self.lineages_fixed.columns = self.times[::-1]
        self.lineages_fixed = self.lineages_fixed.astype(str)
        self.lineages = self.lineages_fixed.copy()
        self.lineages_fixed = self.check_lineages(self.lineages_fixed, interpolation=False, verbose = False)
        self.buildTree()

    def check_lineages(self, lineages, interpolation = True, verbose = True):
        '''
        处理None的标签。
        如果interpolation=True,则把'nan'的数值改为其前一个标签
        '''
        if hasattr(self,'adata') & hasattr(self,'day_field') & hasattr(self,'trajectory_label'):
            class_anno = self.adata.obs.groupby(self.day_field)[self.trajectory_label].unique()
            for lin in lineages.index:
                for t in lineages.columns:
                    if lineages.loc[lin, t] not in class_anno[t].tolist():
                        if interpolation:
                            i = lineages.columns.tolist().index(t)
                            if i > 0:
                                text = lineages.loc[lin, lineages.columns[i-1]]
                            else:
                                text = 'nan'
                        else:
                            text = 'nan'
                        if verbose:
                            print('* [%s] in %s of trajectory [%s] is not exit, set as [%s] of %s'%(lineages.loc[lin, t], t,lin,text,lineages.columns[i-1]))
                        lineages.loc[lin, t] = 'nan'
            if interpolation:
                for lin in lineages.index:
                    for i in range(len(lineages.columns)):       
                        t = lineages.columns[i]
                        if lineages.loc[lin,t] == 'nan':
                            val = lineages.loc[lin,:].values
                            if not all(val[i:] == 'nan'):
                                if interpolation:
                                    lineages.loc[lin,t] = lineages.loc[lin,lineages.columns[i-1]]
                                else:
                                    lineages.loc[lin,t] = '%s_%s'%(lin,'other')
        return(lineages)

    def setNoneWaypoints(self, lineages, min_cells = 0):
        '''
        1. 如果一种细胞类型在时间上不连续，不连续的点设为None
        2. 如果一类细胞在某个时间点的数量少于min_cells,设为None
        '''
        if self.adata is not None:
            for i in lineages.index:
                n = core.del_name_number(i)
                ct_days = self.adata.obs[self.day_field][self.adata.obs[self.trajectory_label] == n]
                all_days = np.array(self.adata.obs[self.day_field].unique())
                all_days = all_days[(all_days <= max(ct_days))&(all_days >= min(ct_days))]
                diff_days = set(all_days) - set(ct_days)
                for t in all_days:
                    self.f1 = self.adata.obs[self.trajectory_label]
                    self.f11 = lineages.loc[n, t]
                    self.f2 = self.adata.obs[self.day_field]
                    self.f22 = t
                    # print(type(self.f11))
                    # print(n)
                    # print(lineages)
                    if np.sum((self.adata.obs[self.trajectory_label]==lineages.loc[n, t]) & (self.adata.obs[self.day_field]==t)) < min_cells:
                        lineages.loc[n, t] = 'nan'
                    elif self.trajectory_ds[self.trajectory_ds.obs[self.day_field]==t, n].X.sum() == 0:
                        lineages.loc[n, t] = 'nan'
                    
        return(lineages)

    # 谱系推断

    def findLineages(self):
        mdf = []
        for i in range(self.trajectory_ds.shape[1]):
            name = self.trajectory_ds.var_names[i]
            end_cells = core.del_name_number(name)

            # 累积的轨迹分数
            # df = self.trajectory_ds.obs
            # df['score'] = np.array(self.trajectory_ds[:,self.trajectory_ds.var_names[i]].X).flatten()
            # df = df.groupby([self.day_field,self.trajectory_label]).mean().dropna().reset_index()

            # 相邻时刻的轨迹分数
            if not hasattr(self, 'couples'):
                self.couple_day_pairs()
            if not hasattr(self, 'sums'):
                self.sum_trajecory_score()
            celltypes_in_day = self.adata.obs.groupby(self.day_field)[self.trajectory_label].unique()
            tracks = []
            preset = True
            for ti in range(len(self.times)):
                # sum_score = df[df[self.day_field] == times[ti]].reset_index(drop=True)
                if preset:
                    if end_cells in list(celltypes_in_day[self.times[ti]]):
                        # 固定最后时刻的细胞类型不变
                        tracks.append(end_cells)
                        preset = False
                    else:
                        tracks.append('nan')
                else:
                    # 1. 根据sum_score的最大值筛选t时刻细胞类型
                    sum_score = self.sums[(self.times[ti],self.times[ti-1])][name]
                    idxmax = sum_score.idxmax()
                    type_of_sum_score = idxmax[1]
                    sum_score_of_type = sum_score[idxmax]

                    # 2. 根据pair_score的最大值筛选t时刻细胞类型
                    pair_score = self.couples[(self.times[ti],self.times[ti-1])][tracks[len(tracks)-1]] # 上一时刻的细胞类型
                    type_of_pair_score = pair_score.idxmax()[1]
                    pair_score_of_type = pair_score.max()

                    if sum_score_of_type > pair_score_of_type:
                        #print('sum_score')
                        tracks.append(type_of_sum_score)
                    else:
                        #print('pair_score')
                        tracks.append(type_of_pair_score)

            mdf.append(pd.DataFrame({name:tracks[::-1]}, index = self.times[::-1]).T)
        return(pd.concat(mdf,axis = 0))

    def buildTree(self, rerun = False, interpolation=True):
        if (not hasattr(self, 'lineages')) or rerun:
            self.lineages = self.findLineages()
            # 设置细胞数量太少及不连续的时间点为None，这些信息将用于计算各个轨迹在时间序列上的平均基因表达
            # self.lineages_fixed = self.setNoneWaypoints(lineages, min_cells = 0)
            # self.lineages = self.lineages_fixed.copy()
        # 对部分为nan的时间点进行插值
        if not hasattr(self, 'lineages_fixed'):
            self.lineages_fixed = self.check_lineages(self.lineages, interpolation=False, verbose=False)
        self.lineages = self.check_lineages(self.lineages, interpolation=interpolation)
        # 允许存在多个起点
        if len(self.lineages.iloc[:,0].unique()) > 1:
            self.lineages.insert(0,-999,'pseudo_root')
            self.ignore_1st = True
        else:
            self.ignore_1st = False
        # 建立有向网络
        lineage_anno = {}
        G_raw = nx.DiGraph()
        for lineage in self.lineages.index:
            lineage_anno[lineage] = []
            for i in range(len(self.lineages.columns)-1):
                t = self.lineages.columns[i]
                t_next = self.lineages.columns[i+1]
                n1 = self.lineages.loc[lineage,t]
                n2 = self.lineages.loc[lineage,t_next]
                if (n1 != 'nan') & (n2 != 'nan'):
                    G_raw.add_edge('%s_%s'%(t,n1), 
                                   '%s_%s'%(t_next,n2), char='')
                    lineage_anno[lineage].extend(['%s_%s'%(t,n1), '%s_%s'%(t_next,n2)])
        self.lineage_anno = {i: list(np.unique(lineage_anno[i])) for i in lineage_anno}

        ## 把node转变为数字字符，避免在构建toytree时出问题
        self.label2id = dict(zip(list(G_raw.nodes()), np.arange(len(G_raw.nodes)).astype(str)))
        self.id2label = {self.label2id[i]: i for i in self.label2id}

        # 将networkx转换成toytree对象
        self.root = list(nx.topological_sort(G_raw))[0]
        # self.G_raw = G_raw
        H = nx.relabel_nodes(G_raw, self.label2id)
        tre = toytree.tree(nx2newick(H, root = self.label2id[self.root]), tree_format=1)
        G = newick2nx(tre)
        G = nx.relabel_nodes(G, self.id2label)
        
        pos = tre._coords.get_linear_coords(TreeStyle('d').layout) # style d,c,o
        node_labels = [tre._coords.ttree.idx_dict[i].name for i in range(len(tre._coords.ttree.idx_dict))]
        node_labels = [self.id2label[i] for i in node_labels]
        
        pos = pd.DataFrame(pos, index = node_labels, columns = ['x','y'])
        pos['time'] = [s.split('_')[0] for s in node_labels]
        pos['label'] = [re.sub('%s_'%pos['time'][i],'',node_labels[i]) for i in range(len(node_labels))]
        pos['time'] = pos['time'].values.astype(float)
        
        self.pos = pos
        self.G_raw = G_raw
        self.G = G
        self.tree = tre
        self.add_node_type(self.G_raw, 'type_raw')
        self.add_node_type(self.G, 'type')


    def sum_trajecory_score(self):
        print('Calculating sum of trajectory score for cell types in their timepoints...')
        df = self.trajectory_ds.obs
        times = self.adata.obs[self.day_field].cat.categories.sort_values(ascending=True)
        sums = {}
        for i in range(len(times)):
            t = times[i]
            if t != times[-1]:
                score = []
                for lin in self.trajectory_ds.var_names:
                    df = self.trajectory_ds.obs
                    df['score'] = np.array(self.trajectory_ds[:,lin].X).flatten()
                    df = df.groupby([self.day_field,self.trajectory_label]).mean().dropna().reset_index()
                    sum_score = df[df[self.day_field] == t].reset_index(drop=True)
                    score.append(sum_score.score.values/sum_score.score.values.sum())
                    index = sum_score.loc[:,self.trajectory_label].values
                index = pd.MultiIndex.from_tuples([['day %s'%t, j]  for j in index])
                # columns = pd.MultiIndex.from_tuples([['day %s'%times[i+1], j]  for j in self.trajectory_ds.var_names])
                columns = self.trajectory_ds.var_names
                res = pd.DataFrame(np.vstack(score).T, index = index, columns = columns)
                res = res.applymap(lambda x: '%.2g'%x)
                res = res.astype(float)
                sums[(t,times[i+1])] = res
        self.sums = sums

    def couple_day_pairs(self):
        print('Calculating pair of trajectory score of adjacent timepoints...')
        self.couples = {}
        for p in sorted(self.tmap_model.day_pairs, key = lambda x: x[0]):
            # print('%s-%s'%(p[0],p[1]))
            cp = self.tmap_model.get_coupling(p[0],p[1])
            cp = cp[cp.obs_names.isin(self.adata.obs_names),:]
            cp = cp[:,cp.var_names.isin(self.adata.obs_names)]

            t0_label = self.adata.obs.loc[cp.obs_names, [self.trajectory_label]]
            t1_label = self.adata.obs.loc[cp.var_names, [self.trajectory_label]]
            
            enc_t0 = OneHotEncoder(handle_unknown='ignore')
            enc_t0.fit(t0_label)
            p0 = enc_t0.transform(t0_label).toarray()
            p0 = p0/p0.sum(0)
            
            enc_t1 = OneHotEncoder(handle_unknown='ignore')
            enc_t1.fit(t1_label)
            p1 = enc_t1.transform(t1_label).toarray()
            p1 = p1/p1.sum(0)
            
            X = p0.T @ cp.X @ p1
            X /= X.sum(0)
            index = pd.MultiIndex.from_tuples([['day %s'%p[0], i]  for i in enc_t0.categories_[0]])
            columns = pd.MultiIndex.from_tuples([['day %s'%p[1], i]  for i in enc_t1.categories_[0]])
            res = pd.DataFrame(X, index = index,columns=enc_t1.categories_[0])
            res = res.applymap(lambda x: '%.2g'%x)
            res = res.astype(float)
            self.couples[p] = res

    def save_day_pairs(self, col_normlize = True, folder = 'ot_pairs'):
        '''
        计算相邻两个时间点之间的细胞类型的转移关系
        '''
        try:
            import openpyxl
        except :
            import subprocess
            subprocess.run('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openpyxl', shell=True)
        if not os.path.exists(folder):
            os.makedirs(folder)
        print('OT results are writing to folder %s...'%folder)
        # 1. pair score
        writer1 = pd.ExcelWriter(os.path.join(folder,'pair_score.xlsx'), engine="openpyxl")
        if not hasattr(self, 'couples'):
            self.couple_day_pairs()
        for p in self.couples:
            self.couples[p].to_excel(writer1, sheet_name="%s<-%s"%(p[0],p[1]))
        writer1.close()
        # 2. sum score
        writer2 = pd.ExcelWriter(os.path.join(folder,'sum_score.xlsx'), engine="openpyxl")
        if not hasattr(self, 'sums'):
            self.sum_trajecory_score()
        for p in self.sums:
            self.sums[p].to_excel(writer2, sheet_name="%s<-%s"%(p[0],p[1]))
        writer2.close()

        self.lineages.to_csv(os.path.join(folder,'lineages.csv'))

    def adjust_layout(self):
        conse = {}
        for branch_node in self.branch:
            for leaf_node in self.leaf:
                try:
                    path = nx.shortest_path(self.G,branch_node,leaf_node)
                    label = [self.pos.loc[i,'label'] for i in path]
                    for ct in np.unique(label):
                        for i in range(len(label)-1):
                            if (label[i] == ct) & (label[i+1] == ct):
                                nodes = set([path[i],path[i+1]])
                                name = '%s_%s'%(self.pos.loc[leaf_node,'label'], ct)
                                if name not in conse:
                                    conse[name] = set()
                                conse[name] = conse[name] | nodes
                except:
                    pass
        # 删除不包含分支节点的路径
        conse = {i:list(conse[i]) for i in conse if len(np.intersect1d(self.branch, list(conse[i]))) > 0}

        # 从小到大排序，修改纵坐标
        n_nodes = [len(conse[i]) for i in conse]
        for i in np.argsort(n_nodes):
            # 连续节点对应的id
            idx = conse[list(conse.keys())[i]]
            # 选择最后的节点
            last_node = self.pos.loc[idx,'time'].idxmax()
            # 最后节点的纵坐标
            y_val = self.pos.loc[last_node, 'y']
            # 修改坐标
            self.pos.loc[idx, 'y'] = y_val    

        # 令两个branch之间，或root和branch之间的点与后一个branch的点对齐
        for node1 in self.branch+self.root:
            for node2 in self.branch:
                try:
                    path = nx.shortest_path(self.G,node1,node2)
                    if not len(np.intersect1d(self.branch, path[1:-1]))>0:
                        if node1 in self.root:
                            self.pos.loc[path[:-1], 'y'] = self.pos.loc[node2, 'y']
                        else:
                            self.pos.loc[path[1:-1], 'y'] = self.pos.loc[node2, 'y']
                except:
                    pass

    def collectTrajectoryPath(self, use_loop=False,smooth_curve = False):
        '''
        提取用于绘制轨迹曲线的坐标
        '''
        self.node_label = {}
        for i in self.lineages.index:
            ids = ['%s_%s'%(self.lineages.columns[j], self.lineages.loc[i, :].values[j]) for j in range(self.lineages.shape[1])]
            self.node_label.update(dict(zip(ids,self.lineages.loc[i, :].values)))
        self.verts = {}
        for i in self.leaf:
            traj = get_ancestors(self.tree, self.label2id[i], self.ignore_1st)
            traj = [self.id2label[x] for x in traj]
            lineage_pos = self.pos.loc[traj,:]
            
            # 平滑部分拐点
            degree = [self.G.degree(i) for i in traj]
            idx = np.arange(len(traj))

            # 需要平滑的节点：自由度为2，且前后节点自由度>2。注意，不平滑joint节点
            if use_loop:
                name = 'type_raw'
            else:
                name = 'type'
            for_sm = []
            after_sm = []
            fix_j = [] # 记录已经被用过的节点
            for j in np.arange(len(degree)-2)+1:
                if degree[j] == 2:
                    if (degree[j-1] > 2 or degree[j+1] > 2 or (degree[j+1] == 2 and (traj[j+1] in self.leaf)) or (lineage_pos.iloc[j-1,:][name] =='joint') and (lineage_pos.iloc[j-1,:].y != lineage_pos.iloc[j,:].y)) and (lineage_pos.index[j] not in self.joint) and (lineage_pos.iloc[j,:][name] !='branch'):
                        for_sm.append(idx[(j-1):(j+2)])
                        val = lineage_pos.iloc[:,:2].values[(j-1):(j+2)]
                        if j-1 in fix_j:
                            val[0] = sm_value[-1]
                        if smooth_curve:
                            niter = 5
                        else:
                            niter = 0
                        sm_value = smooth_line(val, niter = niter)
                        sm = pd.DataFrame(sm_value, columns = ['x','y'], index = [traj[j]]*sm_value.shape[0])                        
                        sm['time'] = self.lineages.columns[j]
                        sm['label'] = self.node_label[traj[j]]
                        after_sm.append(sm)
                        fix_j.append(j)

            # 不需要平滑的节点
            not_sm = []
            if len(for_sm) > 1:
                for k in range(len(for_sm)-1):
                    if k == 0:
                        not_sm.append(lineage_pos.iloc[:for_sm[k][1],:])
                    not_sm.append(lineage_pos.iloc[for_sm[k][2]:for_sm[k+1][1],:])
                    if k == len(for_sm)-2:
                        not_sm.append(lineage_pos.iloc[for_sm[k+1][2]:,:])
            elif len(for_sm) == 1:
                not_sm.append(lineage_pos.iloc[:for_sm[0][1],:])
                not_sm.append(lineage_pos.iloc[for_sm[0][2]:,:])
            else:
                not_sm.append(lineage_pos)
            # import pdb;pdb.set_trace()
            not_sm = [check_connection(x) for x in not_sm]
            # 合并两种类型的节点
            f_i = 0; nf_i = 0
            merge_sm = []
            for x in range(len(after_sm) + len(not_sm)):
                if x % 2 == 0:
                    merge_sm.append(not_sm[nf_i])
                    nf_i += 1
                else:
                    merge_sm.append(after_sm[f_i])
                    f_i += 1
            self.verts[i] = pd.concat(merge_sm)
            if self.ignore_1st:
                self.verts[i] = self.verts[i][self.verts[i].label != 'pseudo_root']
                # self.verts[i] = self.verts[i][self.verts[i].x != self.verts[i].x.min()]
            
        # 补充G_raw的连接
        addition_edges = set(self.G_raw.edges())-set(self.G.edges())
        self.addition_verts = {}
        if len(addition_edges)>0:
            for edge in addition_edges:
                df = self.pos.loc[edge,:]
                df.label = df.label[0]
                self.addition_verts[df.label[0]] = df


    # 轨迹推断模块
    def filterScore(self, trajectory_ds, celltypes, terminal_celltypes=None, mutually_exclusion=True, filter_low_score = True):
        trajectory_ds.obs[self.trajectory_label] = self.adata.obs.loc[trajectory_ds.obs_names.values,
                                                                    self.trajectory_label].values
        for i in celltypes:
            try:
                i_index = trajectory_ds.var_names.tolist().index(i)
                score = trajectory_ds.X[:,i_index]
                ## 非同一类型细胞分数设为0
                if mutually_exclusion:
                    score[trajectory_ds.obs[self.trajectory_label].isin(celltypes[~celltypes.isin([i])].values)] = 0
                ## 分数太低设为0 
                if filter_low_score:
                    score[score < np.quantile(score[score > 0], 0.3)] = 0
                ## 对于terminal_celltypes，大于其最大时间点的细胞轨迹分数设为0
                if terminal_celltypes is not None:
                    if type(terminal_celltypes) is str:
                        terminal_celltypes = [terminal_celltypes]
                    for s in terminal_celltypes:
                        if s in self.adata.obs[self.trajectory_label].unique().tolist():
                            t_max = self.adata.obs[self.day_field][self.adata.obs[self.trajectory_label] == s].max()
                            score[trajectory_ds.obs[day_field] > t_max] = 0
                trajectory_ds.X[:,i_index] = score
            except:
                pass
        return(trajectory_ds)

    def inferLineages(self, celltypes, 
                    add_traj = False,
                    min_cells = 10,
                    contamination=0.2, 
                    run_pca=False, 
                    terminal_celltypes=None,
                    mutually_exclusion=True, 
                    filter_low_score = True):
        '''
        '''
        trajectory_ds = core.trajectory_score(self.adata, 
                                        self.tmap_model,
                                        self.day_field,
                                        self.trajectory_label,
                                        celltypes, 
                                        min_cells = min_cells,
                                        contamination=contamination, 
                                        run_pca=False, 
                                        mutually_exclusion=mutually_exclusion, 
                                        filter_low_score = filter_low_score)


        # 过滤细胞分数
        trajectory_ds = self.filterScore(trajectory_ds, celltypes, terminal_celltypes=terminal_celltypes, 
                                         mutually_exclusion=mutually_exclusion, filter_low_score = filter_low_score)

        if add_traj:
            if hasattr(self, 'trajectory_ds'):
                trajectory_ds = core.hstack_adata(self.trajectory_ds,trajectory_ds)

        self.trajectory_ds = trajectory_ds
        self.couple_day_pairs()
        self.sum_trajecory_score()
        self.buildTree(rerun = True)


    def aveGeneExp(self):
        '''
        计算每个谱系的加权平均基因表达（细胞类型+轨迹分数限制）
        '''
        index = self.trajectory_ds.var_names.values
        self.adata = self.adata[self.trajectory_ds.obs_names.values]
        for t in self.lineages.columns: # lineages_fixed
            # 选出t时刻的基因表达值0和轨迹分数
            t_mask = self.adata.obs[self.day_field] == float(t)
            sub_adata = self.adata[t_mask]
            G = sub_adata.X
            t_adata = self.trajectory_ds[t_mask]
            # 将t时刻不属于lineages中细胞类型的分数设为0
            celltype_mask = np.vstack([t_adata.obs[self.trajectory_label].values == i for i in self.lineages[t]]).astype(int).T
            S = np.multiply(np.array(t_adata.X), celltype_mask)
            S = np.where(S.sum(0) == 0, 0, S/S.sum(0))
            # 平均基因表达
            texp = G.T.dot(S).T
            # 基因表达比例
            n_cells = celltype_mask.sum(0)
            G.data = np.full_like(G.data,1)
            prop = np.vstack([G.T.multiply(celltype_mask[:,i]).T.sum(0) for i in range(len(n_cells))])
            prop = np.where(n_cells == 0, 0, prop.T/n_cells).T

            # 细胞占比
            p_cells = n_cells/len(t_mask)

            if len(sub_adata.obsm) > 0:
                obsm = {i:np.dot(sub_adata.obsm[i].T, S).T for i in sub_adata.obsm}
            else:
                obsm = None
            t_adata = anndata.AnnData(texp, 
                                    obs = pd.DataFrame({self.day_field:t,'lineage':index,
                                                        'trajectory':self.lineages[t].values,
                                                        'node':['%s_%s'%(t, i) for i in self.lineages[t].values],
                                                        'n_cells':n_cells,
                                                        'p_cells':p_cells}),
                                    var = self.adata.var,
                                    obsm = obsm)
            t_adata.layers['proportion'] = prop
            if t == self.lineages.columns[0]:
                mean_adata = t_adata
            else:
                mean_adata = mean_adata.concatenate(t_adata)
        mean_adata.obs.drop(columns=['batch'],inplace = True)
        mean_adata.obs.index = mean_adata.obs['node'].values
        
        # 根据细胞类型分开数据
        ct_adata = {}
        for ct in self.lineages.index:
            ct_adata[ct] = mean_adata[mean_adata.obs['lineage'] == ct]
            # if remove_none_times:
            #     used_time = self.lineages.columns.values[self.lineages.loc[ct,:]!='nan']
            #     ct_adata[ct] = ct_adata[ct][ct_adata[ct].obs[self.day_field].isin(used_time)]
            # ct_adata[ct].obs.reset_index(drop=True, inplace=True)
            ct_adata[ct].obs.index = ct_adata[ct].obs['node'].values
            ct_adata[ct].obs.drop(columns=['node'],inplace = True)
            # 计算高变异基因
            sc.pp.highly_variable_genes(ct_adata[ct], span=1, n_bins = 100, 
                                        min_mean=0.0125, max_mean=2, min_disp=1.5)
            blacklist = findBlackList(ct_adata[ct].var_names)
            ct_adata[ct].var.loc[ct_adata[ct].var_names.isin(blacklist),'highly_variable'] = False
        self.adata_ave = ct_adata
        self.mean_adata = mean_adata[~mean_adata.obs.index.duplicated()]
        self.mean_adata.obs.drop(columns=['lineage','node'],inplace = True)

    # 平滑基因表达
    def smoothExprs(self,
                    lineages,
                    genes,
                    new_samples=None,
                    n_splines=None,
                    spline_order = 3, 
                    max_workers = None):
        if type(lineages) == str:
            lineages = [lineages]
        smooth_exp = {}
        for lin in lineages:
            adata = self.adata_ave[lin]
            if genes is None:
                genes = adata.var_names.values
            else:
                genes = np.unique(np.array(genes))
                genes = genes[pd.Series(genes).isin(adata.var_names.values)]
            x = np.array(adata.obs[self.day_field])
            if new_samples is None:
                x_pred = self.lineages.columns.values
                x_pred = x_pred[x_pred != -999]
            else:
                x_pred = np.linspace(min(x),max(x),100)
            # 如果某个时间点的谱系不存在，对这个时间点做插值处理
            mask = ~((adata.X.sum(1)==0) | (self.lineages.loc[lin,:] == 'nan').values)
            Y_pred = batchSmooth(adata[mask,genes].X, x = x[mask], x_pred = x_pred)
            smooth_exp[lin] = anndata.AnnData(X = Y_pred, var = adata.var.loc[genes,:], 
                                            obs = pd.DataFrame({self.day_field:x_pred,'lineage':lin}))
        return(smooth_exp)


    # 差异表达基因模块
    # 计算branch和leaf节点的差异表达基因


    def check_for_deg_calculation(self, adata, min_cells = 3):
        '''
        检查adata是否满足计算DEG的要求
        '''
        cell_counts = adata.obs.groupby(self.trajectory_label)[self.trajectory_label].count()
        cell_counts_filter = cell_counts[cell_counts > min_cells]
        if len(cell_counts)>1:
            if len(cell_counts) != len(cell_counts_filter):
                if len(cell_counts_filter)>1:
                    adata = adata[adata.obs[self.trajectory_label].isin(cell_counts_filter.index.values)]
                else:
                    return(None)
            return(adata)
        else:
            return(None)
    ## branch
    def FindBranchMarkers(self, radius = 1, save = None, **kwargs):
        '''
        计算每个分支节点及其前后1个节点范围内的所有节点对应细胞类型的差异表达基因
        radius: 以分支节点为中心，以radius为半径搜索节点，计算这些节点对应的细胞类型的DEG
        '''
        nodes = self.pos.index[self.pos['type_raw'] == 'branch']
        degs = {}
        if save is None:
            save = 'branch_deg.xlsx'
        writer = pd.ExcelWriter(save, engine="openpyxl")
        for n in nodes:
            print('Calculate DEGs for branch point [%s]'%n)
            related_nodes = [i for i in nx.ego_graph(self.G_raw, n, radius=radius, undirected = True)]
            sub_pos = self.pos.loc[related_nodes,:]
            times = sub_pos.time.values
            labels = sub_pos.label.values
            sub_adata = self.adata[self.adata.obs[self.day_field].isin(times) & self.adata.obs[self.trajectory_label].isin(labels)]
            sub_adata = self.check_for_deg_calculation(sub_adata, min_cells = 3)
            if sub_adata is not None:
                degs[n] = FindAllMarkers(sub_adata, self.trajectory_label, **kwargs)
                degs[n].to_excel(writer, sheet_name=n)
        writer.close()
        print('Save to %s'%save)
        return(degs)

    def FindLeafMarkers(self, save = None, **kwargs):
        '''
        计算每个分支节点及其前后1个节点范围内的所有节点对应细胞类型的差异表达基因
        '''
        if save is None:
            save = 'leaf_deg.xlsx'
        
        print('Calculate DEGs for leaf nodes...')
        nodes = self.pos.index[self.pos['type_raw'] == 'leaf']
        sub_pos = self.pos.loc[nodes,:]
        times = sub_pos.time.values
        labels = sub_pos.label.values
        sub_adata = self.adata[self.adata.obs[self.day_field].isin(times) & self.adata.obs[self.trajectory_label].isin(labels)]
        sub_adata = self.check_for_deg_calculation(sub_adata, min_cells = 3)
        if sub_adata is not None:
            writer = pd.ExcelWriter(save, engine="openpyxl")
            degs = FindAllMarkers(sub_adata, self.trajectory_label, **kwargs)
            degs.to_excel(writer, sheet_name='leafs')
            writer.close()
        else:
            print('The number of available leaf nodes is less than two.')
        print('Save to %s'%save)
        return(degs)



    # 绘图模块
    def plotTree(self, 
                 adjust_layout = True,
                 smooth_curve = True,
                 cell_type_colors = None, 
                 xlabel = 'Time', 
                 leaf_label = 'lineage', 
                 bg_color = '#F5F5F5',
                 grid_color = '#000000',
                 show_loop = False,
                 linewidth=4,
                 fontsize = 15, 
                 text_pos_adjust=0.4,
                 pseudo_path_nodes = None,
                 show_grid = True, 
                 show_legend = True,
                 title = None,
                 bbox_to_anchor=(1.2, 1.05),
                 figsize = (5,5), 
                 ax = None, 
                 **kwargs):
        '''
        绘制谱系树的结构
        '''
        if pseudo_path_nodes is not None:
            if isinstance(pseudo_path_nodes, str):
                pseudo_path_nodes = [pseudo_path_nodes]
            else:
                pseudo_path_nodes = list(pseudo_path_nodes)
            for node in pseudo_path_nodes:
                if node in self.branch:
                    pseudo_path_nodes += list(self.G.predecessors(node))
        else:
            pseudo_path_nodes = []
        # 生成谱系树轨迹
        if adjust_layout:
            self.adjust_layout()
        self.collectTrajectoryPath(smooth_curve=smooth_curve,use_loop=show_loop)
        if ax is None:
            fig, ax = plt.subplots(figsize =figsize)
        self.show_loop = show_loop
        # 绘制G_raw中非tree结构上的连接
        # 记录已经标记过的标签
        exit_labels = []
        if show_loop:
            name = 'type_raw'
            for ct in self.addition_verts:
                used_verts = self.addition_verts[ct]
                l = self.addition_verts[ct].label[0]
                if cell_type_colors is not None:
                    if l not in exit_labels:
                        label = l
                    else:
                        label = None
                    if l in cell_type_colors:
                        ax.plot(used_verts.x, used_verts.y, c=cell_type_colors[l], label = label,  zorder = 1, **kwargs)
                        exit_labels.append(l)
                    else:
                        if 'Others' not in exit_labels:
                            label = 'Others'
                            exit_labels.append('Others')
                        else:
                            label = None
                        ax.plot(used_verts.x, used_verts.y, c=bg_color, label = label, **kwargs)        
        else:
            name = 'type'
        
        exist_verts = []
        # 各个分支出现的最早时间，画图时先画后出现的分支，再画早出现的分支
        early_time = self.pos.groupby('label')['time'].min().sort_values(ascending = False)
        plot_zorder = pd.Series(np.arange(len(early_time))+2, index = early_time.index)
        for ct in self.verts:
            # 已经画过的轨迹不重复画
            current_verts = list(zip(self.verts[ct].x,self.verts[ct].y))
            mask = ~pd.Series(current_verts).isin(exist_verts).values
            first_true = list(mask).index(True)
            mask[first_true-1] = True
            used_verts = self.verts[ct][mask]

            for i in used_verts.label.unique():
                sub_verts = used_verts[used_verts.label == i]
                c = bg_color
                label = None
                if cell_type_colors is not None:
                    if i not in exit_labels:
                        label = i
                    if i in cell_type_colors:
                        c=cell_type_colors[i]
                        # ax.plot(sub_verts.x,
                        
                        # sub_verts.y, c=cell_type_colors[i], label = label, linestyle=linestyle, zorder = plot_zorder[i], **kwargs)
                        exit_labels.append(i)
                    else:
                        if 'Others' not in exit_labels:
                            label = 'Others'
                            exit_labels.append(label)

                if len(np.intersect1d(pseudo_path_nodes, sub_verts.index.values)) > 0:
                    solid, dash = get_split_index_helper(sub_verts.index.values, pseudo_path_nodes)
                    solid = [np.unique([max(x[0]-1,0)]+x) for x in solid]
                    dash = [np.unique([max(x[0]-1,0)]+x) for x in dash]
                    for idx in solid:
                        ax.plot(sub_verts.x[idx], sub_verts.y[idx], c=c, linewidth=linewidth, linestyle='-', label=label, zorder = plot_zorder[i], **kwargs)

                    for idx in dash:
                        ax.plot(sub_verts.x[idx], sub_verts.y[idx], c=c, linewidth=linewidth, alpha=0.2, linestyle=(0,(1,1)), label=label, zorder = plot_zorder[i], **kwargs)

                else:
                    ax.plot(sub_verts.x, sub_verts.y, c=c, linewidth=linewidth, linestyle='-', label=label, zorder = plot_zorder[i], **kwargs)

                # 记录已经画过的轨迹
                if sub_verts.shape[0] > 1:
                    exist_verts.extend(list(zip(sub_verts.x,sub_verts.y)))    
            leaf = self.verts[ct][self.verts[ct].index.isin(self.leaf)]
            leaf.sort_values('x', ascending=False,inplace = True)
            # 分支标签
            if ct in self.pos.index[self.pos[name] == 'leaf'].tolist():
                if leaf_label == 'lineage':
                    for l in self.lineage_anno:
                        if ct in self.lineage_anno[l]:
                            text = core.del_name_number(l)
                    ax.text(leaf.x.tolist()[0]+text_pos_adjust,leaf.y.tolist()[0],text,ha = 'left',va='center',fontsize = fontsize)
                if leaf_label == 'leaf':
                    ax.text(leaf.x.tolist()[0]+text_pos_adjust,leaf.y.tolist()[0],leaf.label.tolist()[0],ha = 'left',va='center',fontsize = fontsize)
            ax.set_xticks(np.linspace(self.pos.x.min(),self.pos.x.max(),self.lineages.shape[1]))
            ax.set_xticklabels(self.lineages.columns.values, rotation=90)
            ax.tick_params(direction='out', length=4, width=1, bottom=True)
            ax.spines[['left','right', 'top']].set_visible(False)
            ax.set_yticks([])
            ax.grid(show_grid,color=grid_color)
            ax.set_xlabel(xlabel)
        root = self.pos.loc[self.root,:]
        for i in range(len(root)):
            ax.text(root.iloc[i,:].x-text_pos_adjust,root.iloc[i,:].y,root.iloc[i,:].label,ha = 'right',va='center',fontsize = fontsize)
        if show_legend & (cell_type_colors is not None):
            ax.legend(bbox_to_anchor=bbox_to_anchor,frameon=False)
        ax.set_title(title)
        self.ax = ax
        self.fig = fig

    def plotNodes(self,                   
              root_color = '#CE373A',
              branch_color = '#2872C4',
              leaf_color = '#1A9B51',
              joint_color = '#EDB113',
              **kwargs):

            scatter_zorder = max([_.zorder for _ in self.ax.get_children()])+1
            
            if self.show_loop:
                name = 'type_raw'
            else:
                name = 'type'
            root = self.pos[self.pos[name] == 'root']
            branch = self.pos[self.pos[name] == 'branch']
            leaf = self.pos[self.pos[name] == 'leaf']
            joint = self.pos[self.pos[name] == 'joint']
            
            if len(root) == 0:
                root_color = None
            if len(branch) == 0:
                branch_color = None
            if len(leaf) == 0:
                leaf_color = None
            if len(joint) == 0:
                joint_color = None
            
            if root_color is not None:
                self.ax.scatter(root.x, root.y, c = root_color, label = 'Root', zorder = scatter_zorder, **kwargs)# edgecolor , linewidth,
            if branch_color is not None:
                self.ax.scatter(branch.x, branch.y, c = branch_color, label = 'Branch points', zorder = scatter_zorder, **kwargs)
            if leaf_color is not None:
                self.ax.scatter(leaf.x, leaf.y, c = leaf_color, label = 'Leaf',zorder = scatter_zorder, **kwargs)
            if joint_color is not None:
                self.ax.scatter(joint.x, joint.y, c = joint_color, label = 'Joint',zorder = scatter_zorder, **kwargs)

    def plotExprs(self,
                gene, center = None,
                cmap = None,
                vmin = None,
                vmax = None,
                sizeby = 'proportion',
                size = 1,
                smallest_size = 0.05,
                edgecolor = '#000000',
                linewidth = 1,
                show_legend = True,
                num_of_legend_dots = 4,
                legend_pos = (1.35, 0.1),
                colorbar_pos=[1.5, 0.1, 0.03, 0.8],
                **kwargs):
        '''
        在tree上绘制基因表达
        以center为中心，搜索所有经过center的节点
        '''

        scatter_zorder = max([_.zorder for _ in self.ax.get_children()])+1
        if center is None:
            path = self.pos.index.values
        else:
            path = core.path_search(self.G_raw, self.root, center)
            path.extend(core.dfs_search(self.G_raw, center))
        
        if cmap is None:
            colormap = mymap
        else:
            colormap = plt.cm.get_cmap(cmap)
        
        data = self.mean_adata[path,gene]
        values = data.X.toarray().flatten()
        

        if sizeby is None:
            s = size
        elif sizeby == 'proportion':
            props = data.layers['proportion'].toarray().flatten() + smallest_size
            s = props*size
        elif sizeby == 'n_cells':
            s = data.obs['n_cells'].values*size
        elif sizeby == 'p_cells':
            s = data.obs['p_cells'].values*size

        if vmin is None:
            vmin = min(self.mean_adata[:,gene].X)
        if vmax is None:
            vmax = max(self.mean_adata[:,gene].X)
        pos = self.pos.loc[path,['x','y']]
        plot = self.ax.scatter(pos.x, pos.y, c = values, s = s,
                               zorder = scatter_zorder,
                               vmin = vmin, vmax = vmax, edgecolor=edgecolor, linewidth=linewidth,
                               cmap=colormap, **kwargs)# edgecolor , linewidth,
        handles, labels = plot.legend_elements(prop="sizes",num=61,alpha=1,func = lambda x:x/size-smallest_size)
        
        # 选择合适的legend
        if show_legend:
            def findlegend(handles, labels, n):
                interval = int(len(labels) / (n - 1))
                used_legends = [0] + [(i+1) * interval for i in range(n-2)] + [len(labels)-1]
                return([handles[i] for i in used_legends], [labels[i] for i in used_legends])
            
            handles, labels = findlegend(handles, labels, num_of_legend_dots)

            self.ax.legend(handles, labels,bbox_to_anchor=legend_pos,frameon=False,
                        loc="lower left", title="Fraction of cells")
        self.fig.colorbar(plot, ax=self.ax, cax=self.fig.add_axes(colorbar_pos), shrink=0.2)
        


    def trajectoryPlot(self,
                    show_lineages = None,
                    style = 'merge',
                    basis = 'X_umap',
                    day_field = 'day',
                    linestyle = 'solid',
                    edgecolors = '#000000',
                    highlight_of_color = None,
                    color = None,
                    color_map = 'plasma',
                    cmap_time = 'Spectral_r',
                    trace_color = None,
                    bg_color = '#EBEBEB',
                    min_quantile = 0.01,
                    linecolor = None,
                    dotcolor = None,
                    add_outline = True,
                    size = 2,
                    edgewidth = 2,
                    point_size=1,
                    line_alpha = 1,
                    point_alpha = 1,
                    ncol = 3,
                    linewidths = 1,
                    show_legend=True,
                    bbox_to_anchor=(1.2, 1.05),
                    figsize = (8,8)):
        '''
        在2D layout展示轨迹变化
        '''
        ## 初始化一些变量
        uns = self.adata._uns.copy()
        self.adata = self.adata[self.trajectory_ds.obs_names.values]
        self.adata._uns = uns
        obs = self.adata.obs
        obsm = pd.DataFrame(self.adata.obsm[basis][:,:2], index = obs.index.values)
        if show_lineages is None:
            lineages = self.lineages
        else:
            if type(show_lineages) is str:
                show_lineages = [show_lineages]
            lineages = self.lineages.loc[show_lineages,:]
        n_lineages = lineages.shape[0]
        ncol = min(n_lineages, ncol)    
        
        # 配置颜色
        ## 时间颜色
        days = obs[self.day_field].cat.categories.tolist()
        if (f'{self.day_field}_colors' in self.adata.uns) & (type(self.adata.uns[f'{self.day_field}_colors']) is not dict):
            c_time = self.adata.uns[f'{self.day_field}_colors']
        else:
            c_time = create_gradient_color(palette = cmap_time,n=len(days))
        time_color = dict(zip(days, c_time))
        self.adata._uns[f'{self.day_field}_colors'] = c_time
        
        ## 细胞类型颜色
        if trace_color is None:
            if f'{self.trajectory_label}_colors' in self.adata.uns:
                labels = obs[self.trajectory_label].cat.categories.tolist()
                trace_color = dict(zip(labels, self.adata.uns[f'{self.trajectory_label}_colors']))
            else:
                labels = obs[self.trajectory_label].unique().tolist()
                trace_color = dict(zip(labels, ['C%s'%i for i in range(len(labels))]))
                self.adata._uns[f'{self.trajectory_label}_colors'] = ['C%s'%i for i in range(len(labels))]     

        if style == 'merge':
            _, ax = plt.subplots(1,1, figsize = figsize)
            for i in range(n_lineages):     
                ## 轨迹线颜色
                if linecolor is None:
                    linecolor_ax = trace_color[core.del_name_number(self.trajectory_ds.var_names[i])]
                else:
                    linecolor_ax = linecolor
                adata_ave = self.adata_ave[lineages.index[i]]
                # 绘制每个lineage在不同时间点之间的连线
                ax.plot(adata_ave.obsm[basis][:,0], 
                        adata_ave.obsm[basis][:,1],
                        lw=linewidths,
                        linestyle=linestyle,
                        alpha = line_alpha,
                        label = lineages.index[i],
                        c=linecolor_ax,
                        zorder = 2)
                for t in days:
                    ## 令点的颜色为lineage的默认颜色
                    if dotcolor is None:
                        dotcolor_ax = trace_color[core.del_name_number(self.trajectory_ds.var_names[i])]
                    elif dotcolor == 'time':
                        dotcolor_ax = time_color[t]
                    else:
                        dotcolor_ax = dotcolor
                    
                    ## 绘制每个lineage在每个时间点的中心点
                    ax.scatter(adata_ave.obsm[basis][adata_ave.obs[self.day_field]==t,0], 
                            adata_ave.obsm[basis][adata_ave.obs[self.day_field]==t,1], 
                            alpha = point_alpha,
                            c=dotcolor_ax,
                            s=point_size,
                            edgecolors = edgecolors,
                            linewidth = edgewidth,
                            zorder=3)
            
            ## 绘制layout
            sc.pl.embedding(self.adata, basis = basis,
                            color = color, 
                            size = size,
                            add_outline = add_outline,
                            ax=ax, frameon = False, show=False)

            ## 细胞类型标签
            if show_legend:      
                ax.legend(bbox_to_anchor=bbox_to_anchor)

        if style == 'split':
            if n_lineages == ncol:
                nc = ncol
                nr = 1
                pp = list(np.arange(ncol))
                rm_pp = pp.copy()
            elif n_lineages > ncol:
                nc = ncol # number of columns
                nr = np.ceil(n_lineages/nc).astype(int) # number of rows
                pp = list(itertools.product(np.arange(nr), np.arange(nc)))
                rm_pp = pp.copy()
            _, axs = plt.subplots(nr,nc, figsize = figsize)
            for i in range(n_lineages):
                if nc == nr == 1:
                    ax = axs
                    rm_pp = None
                elif (nc == 1) | (nr ==1):
                    ax = axs[pp[i]]
                    rm_pp.pop(0)
                else:
                    ax = axs[pp[i][0], pp[i][1]]
                    rm_pp.pop(0)
                if linecolor is None:
                    linecolor_ax = trace_color[core.del_name_number(self.trajectory_ds.var_names[i])]
                else:
                    linecolor_ax = linecolor
                
                adata_ave = self.adata_ave[lineages.index[i]]


                # 绘制每个lineage在不同时间点之间的连线
                ax.plot(adata_ave.obsm[basis][:,0], 
                        adata_ave.obsm[basis][:,1],
                        lw=linewidths,
                        linestyle=linestyle,
                        alpha = line_alpha,
                        c=linecolor_ax,
                        zorder = 2)
                for t in days:
                    ## 令点的颜色为lineage的默认颜色
                    if dotcolor is None:
                        dotcolor_ax = trace_color[core.del_name_number(self.trajectory_ds.var_names[i])]
                    elif dotcolor == 'time':
                        dotcolor_ax = time_color[t]
                    else:
                        dotcolor_ax = dotcolor
                    
                    ## 绘制每个lineage在每个时间点的中心点
                    if isinstance(point_size, dict):
                        point_size_i = point_size[lineages.index[i]]
                        if isinstance(point_size_i, (list, tuple, np.ndarray)):
                            psize = point_size_i[list(days).index(t)]
                        else:
                            psize = point_size_i
                    else:
                        psize = point_size
                    ax.scatter(adata_ave.obsm[basis][adata_ave.obs[self.day_field]==t,0], 
                            adata_ave.obsm[basis][adata_ave.obs[self.day_field]==t,1], 
                            alpha = point_alpha,
                            c=dotcolor_ax,
                            s=psize,
                            edgecolors = edgecolors,
                            linewidth = edgewidth,
                            zorder=3)

                # 把每个时间点分数最大的细胞群体挑选出来展示
                idx = []
                for j in range(adata_ave.shape[0]):
                    idx.append(obs.index.values[(obs[self.day_field] == float(adata_ave.obs[self.day_field][j])) & (obs[self.trajectory_label] != adata_ave.obs.trajectory[j])])
                idx = np.concatenate(idx)

                score = np.array(self.trajectory_ds[:,i].X).flatten()
                if color is None:
                    temp_labs = pd.Series(np.array(obs[self.day_field]), index = obs.index.values)
                    temp_labs[idx] = 0
                    temp_labs[score < np.quantile(score, min_quantile)] = 0 ## 将轨迹分数低于min_quantile的值设为0
                    temp_labs = np.array([str(v) for v in temp_labs])
                    categories=np.concatenate([['0'],[str(v) for  v in days]])
                else:
                    temp_labs = pd.Series(np.array(obs[color]), index = obs.index.values)
                    temp_labs[idx] = 0
                    temp_labs[score < np.quantile(score, min_quantile)] = 0 ## 将轨迹分数低于min_quantile的值设为0
                    temp_labs = np.array([str(v) for v in temp_labs])
                    unique_temp_labs = list(np.unique(temp_labs))
                    categories=np.concatenate([['0'],obs[color].cat.categories[obs[color].cat.categories.isin(unique_temp_labs)]])
                    color_anno = dict(zip(obs[color].cat.categories, self.adata.uns['%s_colors'%color]))
                    c_time = [color_anno[n] for n in categories[1:]]

                ## 调整彩色的点在图层上方，灰色（非轨迹上的细胞）点在图层下方
                pseudo_adata = anndata.AnnData(obs = pd.DataFrame(index = obs.index.values), obsm = self.adata.obsm)
                pseudo_adata.obs['temp_color'] = temp_labs
                if highlight_of_color is not None:
                    if color is not None:
                        mask = obs[color].str.contains(highlight_of_color)
                        pseudo_adata = pseudo_adata[~mask].concatenate(pseudo_adata[mask])
                sub_adata = pseudo_adata[pseudo_adata.obs['temp_color']=='0'].concatenate(pseudo_adata[pseudo_adata.obs['temp_color']!='0'])
                sub_adata.obs['temp_color'] = pd.Categorical(sub_adata.obs['temp_color'], categories=categories)

                ## 时间ID为0的细胞为灰色 #EBEBEB
                sub_adata.uns['temp_color_colors'] = [bg_color] + list(c_time)

                ## 绘制UMAP
                sc.pl.embedding(sub_adata, basis = basis,
                                color = 'temp_color', 
                                size = size,
                                add_outline = add_outline,
                                title=lineages.index[i], #legend_loc ='on data',
                                ax=ax, frameon = False, show=False)

                ## 时间点太多了，这里设置不显示时间legend                
                if color is None:
                    legend=ax.legend('')
                    legend.remove()

            # remove unused subplots
            if not rm_pp == None:
                if len(rm_pp) > 0:
                    for rmplot in rm_pp:
                        if nc == 1 | nr ==1:  
                            axs[rmplot].set_axis_off()
                        else:
                            axs[rmplot[0], rmplot[1]].set_axis_off()   