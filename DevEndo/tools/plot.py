import itertools
import scanpy as sc
import scipy.sparse as sp
import importlib
from scipy.sparse import csr_matrix
from .comparison import nonzero
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import rc_context
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from itertools import combinations

plt.rcParams['ytick.left'] = True  # 设置 y 轴刻度线可见
plt.rcParams['xtick.bottom'] = True  # 设置 x 轴刻度线可见
mpl.rcParams['pdf.fonttype']  = 42

from .core import *
from .comparison import *

colors2 = plt.cm.Reds(np.linspace(0.2, 1, 100))
colorsComb = np.vstack([colors2])
mymap_reds = colors.LinearSegmentedColormap.from_list('', colorsComb)

colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,1))
colors2 = plt.cm.Reds(np.linspace(0.0, 1, 99))
colorsComb = np.vstack([colors1, colors2])
mymap = colors.LinearSegmentedColormap.from_list('grey-red', colorsComb)


colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,1))
colors2 = plt.cm.Reds(np.linspace(0.0, 0.9, 99))
colorsComb = np.vstack([colors1, colors2])
mymap_gr = colors.LinearSegmentedColormap.from_list('grey-red', colorsComb)


colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,1))
colors2 = plt.cm.Blues(np.linspace(0.0, 1, 99))
colorsComb = np.vstack([colors1, colors2])
mymap_gb = colors.LinearSegmentedColormap.from_list('grey-blues', colorsComb)

colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,1))
colors2 = plt.cm.Purples(np.linspace(0.0, 1, 99))
colorsComb = np.vstack([colors1, colors2])
mymap_gp = colors.LinearSegmentedColormap.from_list('grey-Purples', colorsComb)

colors1 = plt.cm.Greys_r(np.linspace(0.8,0.9,1))
colors2 = plt.cm.Greens(np.linspace(0.0, 1, 99))
colorsComb = np.vstack([colors1, colors2])
mymap_gg = colors.LinearSegmentedColormap.from_list('grey-green', colorsComb)

colors1 = [1.0, 1.0, 1.0, 1.0]
colors2 = plt.cm.Reds(np.linspace(0.0, 1, 100))
colorsComb = np.vstack([colors1, colors2])
mymap_wr = colors.LinearSegmentedColormap.from_list('white-red', colorsComb)

mymap_bwr = colors.LinearSegmentedColormap.from_list('bwr', ['#424D9F','#ffffff','#E42A1E'])


def create_dot_size_legend(val, 
                           ax,
                           times = 1,
                           interval = 2,
                           title=None,
                           color='none',
                           edgecolor='black'):
    """
    创建散点图中点大小的图例。

    参数：
    val：一个包含数据值的可迭代对象，用于确定点大小的范围。
    ax：matplotlib 的坐标轴对象，用于在该坐标轴上创建图例。
    interval（可选，默认值为 2）：确定图例中不同点大小的步长间隔。
    title（可选）：图例的标题。
    color（可选，默认值为 'none'）：散点的填充颜色。
    edgecolor（可选，默认值为 'black'）：散点的边缘颜色。

    """
    size_factor = 10**(count_digits(max(val))-1)
    vmax = int(np.ceil(max(val)/size_factor))
    vmin = int(np.floor(min(val)/size_factor))
    steps = np.arange(vmin,vmax+1,interval)
    handles = []
    for step in steps:
        handle = plt.scatter([], [], s=step*size_factor*times+1, color=color, edgecolor=edgecolor)
        handles.append(handle)
    ax.legend(handles=handles, labels = [str(round(i,1)) for i in steps*size_factor], 
              loc="upper left",title=title,frameon=False)
    ax.axis('off')
    ax.set_position([0,1,0.2,1])
    



def get_group_colors(adata, key, colors = None):
    '''
    Make a dict of colors for adata.obs[key].
    '''
    if colors is None:
        default_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] # matplotlib 默认循环颜色
    else:
        default_colors = colors
    cats = factorOrder(adata.obs[key])
    if '%s_colors'%key in adata.uns:
        group_color = dict(zip(cats, adata.uns['%s_colors'%key]))
    else:
        gen_colors = [default_colors[i%len(default_colors)] for i in range(len(cats))]
        group_color = dict(zip(cats, gen_colors))
    return(group_color)


def stage_colors(adata, stage_key,cmap = 'Spectral_r', n = None):
    '''
    Assign stage colors to stage_key in adata
    '''
    if n is None:
        nc = len(adata.obs[stage_key].unique())
    else:
        nc = n
    adata.uns['%s_colors'%stage_key] = create_gradient_color(sns.color_palette(cmap, nc), nc)


def bestcolors(n=2,contrast='high',show=False):
    """
    Return preset colors
    --------

    n: number of colors
    contrast: high, mid, low
    show: show color examples
    """
    if n == 2:
        if contrast == 'high':
            r = ['#EE5A4D','#303854']
        if contrast == 'mid':
            r = ['#117D8A','#F26061']
        if contrast == 'low':
            r = ['#77C3A5','#FBBA00']

    if n == 3:
        if contrast == 'high':
            r = ['#CF3A36','#3B7C70','#4063A3']
        if contrast == 'mid':
            r = ['#FF7114','#159C95','#2473A2']
        if contrast == 'low':
            r = ['#EE404E','#FBBB48','#0099D5']

    if n == 4:
        if contrast == 'high':
            r = ['#659A32','#046E8F','#F9C22E','#DD4124']
        if contrast == 'mid':
            r = ['#DD5129','#0F7BA2','#43B284','#FAB255']
        if contrast == 'low':
            r = ['#3FB8AF','#69D2E7','#EBA42B','#DA7698']
    if n == 5:
        if contrast == 'high':
            r = ['#DB3D06','#FCD723','#8EB035','#7DB3E2','#3D3E52']
        if contrast == 'mid':
            r = ['#CC3D24','#F3C558','#6DAE90','#30B4CC','#004F7A']
        if contrast == 'low':
            r = ['#E01A4F','#F15946','#F9C22E','#53B3CB','#7DCFB6']
            r = ['#F26386','#F588AF','#A4D984','#FCBC52','#FD814E']

    if n == 6:
        if contrast == 'high':
            r = ['#9F248F','#FFCE4E','#017A4A','#F9791E','#244579','#C6242D']
        if contrast == 'mid':
            r = ['#DC0000','#FF6400','#FEB24C','#64AA00','#506EBE','#783C8C']
        if contrast == 'low':
            r = ['#FF9062','#FD6598','#CB64C0','#3294DD','#7DCFB6','#D0EB60']

    if n == 7:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#86C4E1','#4D6AB8', '#6F45A2', '#73C3A4', '#CC2E11','#D65E9B', '#EA8800']
        if contrast == 'low':
            r = ['#EFB279','#F7E69E','#3D7BB7','#B98FC1','#4BAA99','#C96678','#F59EB4']

    if n == 8:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#4C5454','#DC0000','#FF6400','#FFDC14','#64AA00','#506EBE','#8C3C78','#FF96C8']
        if contrast == 'low':
            r = ['#61BEA4','#B6E7E0','#AA3F5D','#DAA5AC','#98A54F','#2E92A2','#FFB651','#D85A44']

    if n == 9:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#8ED3C7','#FD7F72','#D9D51B','#BEBADB','#7DB0D1','#D3934D','#8B43AB','#F3CF3E','#52BD7E']

    if n == 10:
        if contrast == 'high':
            r = ['#3F67B6','#F48023','#9D368B','#D48892','#9E4027','#5689C8','#C92E2C','#84C246','#66448D','#DFB149']
        if contrast == 'mid':
            r = ['#D31F28','#272E6A','#228843','#87298B','#F17C2A','#FCE300','#879ED0','#BE6CAB','#D7A767','#5F8C96']
        if contrast == 'low':
            r = ['#CF87B9','#F6B1D0','#FFE07B','#DAA882','#10A597','#558AC8','#EF473A','#A7ADDA','#83CBF1','#EF7D21']

    if n == 11:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#FF4D6F','#579EA4','#DF7713','#F9C000','#86AD34','#5D7298','#81B28D','#7E1A2F','#2D2651','#C8350D','#BD777A']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#']

    if n == 12:
        if contrast == 'high':
            r = ['#5F8C96','#DDAB55','#D92330','#CA7D87','#3369AF','#73B55C','#272E6A','#ED7D31','#9A3986','#54BFE3','#998DB9','#9E4424']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#']

    if n == 13:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#']

    if n == 14:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#196CCD','#C751D5','#DE292E','#F2BD4C','#FD7C08','#478EDB','#8FCBEB','#7E40D1','#9B67BC','#9E3D19','#D78C8C','#7ACC5D','#633F8F','#D9AE4F']


    if n == 15:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']


    if n == 16:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']


    if n == 17:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#1D9D78','#E25D00','#59969B','#FE9A90','#ED73BE','#E4A908','#9A4EA4','#A6CFE3','#14B7C8','#E32B85','#FF8100','#746FB1','#B6B929','#B55626','#FD4E33','#3DBD44','#A27518']

    if n == 18:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']

    if n == 19:
        if contrast == 'high':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']

    if n == 20:
        if contrast == 'high':
            r = ['#3E51A8', '#5FBDB0', '#545B65', '#C65E5E', '#D93786', '#F3A83D', '#7E56C2', '#B63A3A', '#D17B46', '#4A8EC6',
                 '#F39237', '#66A966', '#D68888', '#FAD337', '#66C966', '#94C9E5', '#009A90', '#D6ACEF', '#D1A33C', '#D1B99B']
        if contrast == 'mid':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
        if contrast == 'low':
            r = ['#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#','#']
    return(r)
# def bestcmap():

def highLightGroup(adata, groupby, label=None,label_color = None,
                   target_groups=None,
                   basis = 'X_umap',
                   target_color = '#2174AF',
                   other_color = '#D4D4D4',
                   frameon=None,
                   add_outline=False,
                   ncol = 3,
                   figsize = (8,8)):
    """
    Generate a grid of 2D embedding plots highlighting specific cell groups while dimissing others.
    
    Creates a panel of UMAP (or other embedding) plots where each subplot highlights one target group
    with distinct coloring, while all other cells are shown in a muted color. Optionally overlays
    secondary label colors on the highlighted group for additional annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing single-cell data with precomputed embeddings in `.obsm`
    groupby : str
        Column name in `adata.obs` defining the primary cell groups to highlight (e.g., 'cell_type', 'time_point')
    label : str, optional
        Secondary annotation column in `adata.obs` to color the highlighted group (e.g., 'cluster', 'condition');
        if None, only target/other coloring is used
    label_color : dict, optional
        Custom color mapping dictionary for the `label` categories (keys = category names, values = hex/rgb colors);
        if None, uses existing color palette from `adata.uns['{label}_colors']`
    target_groups : list or str, optional
        Specific groups from `groupby` column to highlight (single group as str or multiple as list);
        if None, uses all unique categories in `adata.obs[groupby]`
    basis : str, default='X_umap'
        Key in `adata.obsm` containing the 2D embedding coordinates (e.g., 'X_tsne', 'X_pca')
    target_color : str, default='#2174AF'
        Hex/rgb color for highlighting target group cells when `label=None`
    other_color : str, default='#D4D4D4'
        Hex/rgb color for non-target (background) cells across all plots
    frameon : bool, optional
        Whether to draw a frame around each subplot; uses matplotlib default if None
    add_outline : bool, default=False
        Whether to add outline around cell clusters (scanpy's add_outline parameter)
    ncol : int, default=3
        Maximum number of columns in the plot grid
    figsize : tuple, default=(8,8)
        Overall figure size (width, height) in inches

    Returns
    -------
    numpy.ndarray
        Array of matplotlib Axes objects for the generated subplots, enabling further customization
    
    Notes
    -----
    - Embedding coordinates must be precomputed and stored in `adata.obsm[basis]`
    - Target groups are displayed in a grid layout with maximum `ncol` columns
    - All subplots remove axis ticks/labels and legends for cleaner visualization
    - Temporary AnnData objects are created for each subplot to handle dynamic coloring
    
    Examples
    --------
    >>> # Highlight all cell types with default coloring
    >>> axs = highLightGroup(adata, groupby='cell_type', basis='X_umap')
    
    >>> # Highlight specific time points with cluster labels and custom colors
    >>> color_map = {'Stem': '#FF5733', 'Progenitor': '#33FF57', 'Differentiated': '#3357FF'}
    >>> axs = highLightGroup(
    ...     adata,
    ...     groupby='time_point',
    ...     label='cell_state',
    ...     label_color=color_map,
    ...     target_groups=['Day3', 'Day7'],
    ...     ncol=2,
    ...     figsize=(10,5)
    ... )
    """
    if target_groups is None:
        target_groups = pd.Categorical(adata.obs[groupby]).categories.tolist()
    else:
        if isinstance(target_groups,str):
            target_groups = [target_groups]        

    if label_color is None:
        label_color = dict(zip(adata.obs[label].cat.categories.tolist(), adata.uns['%s_colors'%label]))
    # Plot
    n = len(target_groups)
    ncol = min(n, ncol)
    if n == ncol:
        nc = ncol
        nr = 1
        pp = list(np.arange(ncol))
        rm_pp = pp.copy()
    elif n > ncol:
        nc = ncol # number of columns
        nr = np.ceil(n/nc).astype(int) # number of rows
        pp = list(itertools.product(np.arange(nr), np.arange(nc)))
        rm_pp = pp.copy()    

    fig, axs = plt.subplots(nr,nc,figsize=figsize)
    for i in np.arange(n):
        if nc == nr == 1:
            ax = axs
            rm_pp = None
        elif (nc == 1) | (nr ==1):
            ax = axs[pp[i]]
            rm_pp.pop(0)
        else:
            ax = axs[pp[i][0], pp[i][1]]
            rm_pp.pop(0)
        target_group = target_groups[i]
        new_id = np.array(['_Other']*adata.shape[0],dtype = object)
        new_id[adata.obs[groupby] == target_group] = target_group
        new_id[adata.obs[groupby] != target_group] = '_Other'
        temp = anndata.AnnData(csr_matrix(adata.shape), obs = pd.DataFrame({'hl_%s'%target_group: new_id}), obsm = adata.obsm)

        
        
        if label is not None:
            temp.obs['label'] = np.array(adata.obs[label])
            temp1 = temp[temp.obs['hl_%s'%target_group] != target_group]
            temp2 = temp[temp.obs['hl_%s'%target_group] == target_group]
            temp2.obs['hl_%s'%target_group] = np.array(temp2.obs['label'])
        else:
            temp1 = temp[temp.obs['hl_%s'%target_group] != target_group]
            temp2 = temp[temp.obs['hl_%s'%target_group] == target_group]
        temp = temp1.concatenate(temp2)
        if label is not None:
            # print(list(np.unique(temp.obs['hl_%s'%target_group]))+ ['_Other'])
            temp.obs['hl_%s'%target_group] = pd.Categorical(np.array(temp.obs['hl_%s'%target_group]), list(np.unique(temp2.obs['hl_%s'%target_group]))+ ['_Other'])
            temp.uns['hl_%s_colors'%target_group] = [label_color[i] for i in temp.obs['hl_%s'%target_group].cat.categories if i != '_Other'] + [other_color]
        else:
            temp.obs['hl_%s'%target_group] = pd.Categorical(np.array(temp.obs['hl_%s'%target_group]), [target_group, '_Other'])
            temp.uns['hl_%s_colors'%target_group] = [target_color, other_color]
        sc.pl.embedding(temp, basis = basis, color = 'hl_%s'%target_group, title = target_group, 
                        frameon = frameon, add_outline=add_outline,
                        show = False, ax=ax)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.get_legend().remove()
    return(axs)
    
def show_bestcolors(r, n_samples=200, fontsize = 8, rotation = 90, figsize = (8,2)):
    fig, ax = plt.subplots(1, 3,figsize = figsize)
    # 1. barplot
    ax[0].bar(r, 1, color = r)
    ax[0].grid(False)
    ax[0].set_yticks([])
    ax[0].set_xticklabels(r, rotation=rotation, ha="center",fontsize = fontsize)
    # 2. dotplot
    X, _ = make_blobs(n_samples=n_samples, centers=len(r), cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=len(r),max_iter = 10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    for i in range(len(r)):
        ax[1].scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], c=r[i], s=20, label = r[i])

    ax[1].grid(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # 3. curvplot
    x = np.arange(10)
    y = 2.5 * np.sin(x / 20 * np.pi)
    for i in range(len(r)):
        ax[2].plot(x, y-i, c = r[i], label = r[i])
    ax[2].grid(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.show()

def LerpColour(c1,c2,t):
    return (c1[0]+(c2[0]-c1[0])*t, c1[1]+(c2[1]-c1[1])*t, c1[2]+(c2[2]-c1[2])*t)
def create_gradient_color(colors_list, 
                          step, 
                          to_hex=True
                          ):
    """
    Generated gradient color
    
    Parameters:
    ----------
    colors_list: colors list of "rgb" or "hex"
    step: The numbers of colors are generated
    to_hex: rgb converted to hex
    ---------
    Usage:
    -----
    import sc_py_function_set as mysc
    col_list = di.create_gradient_color(sns.color_palette("tab20", 20), 50)
    col_list = np.array(col_list)
    -----
    """
    if '#' in colors_list[0]:
        rgb = []
        for h in colors_list:
            rgb.append(mcolors.to_rgb(h))
    else:
        rgb = colors_list
    st = len(rgb)-2
    gradient = []
    for i in range(st+1):
        for j in range(step):
            gradient.append(LerpColour(rgb[i],rgb[i+1],j/step))
    gradient = np.array(gradient)
    rgb_list = []
    for i in range(0, len(gradient)-1, st+1):
        rgb_list.append((np.sum(gradient[i:i+st+1],axis=0)/(st+1)).tolist())
    if to_hex:
        rgb_or_hex=[]
        for rth in rgb_list:
            rgb_or_hex.append(mcolors.to_hex(rth))
    else:
        rgb_or_hex = rgb_list
    return rgb_or_hex




def plotTrajectory(adata, source, 
                   basis = 'X_umap',
                   color = None,
                   style = 'merge',
                   linestyle='solid',
                   linewidths = 5,
                   line_alpha=1,
                   point_size = 100,
                   point_alpha=1,
                   edgecolors = '#000000',
                   edgewidth = 2,
                   size = None,
                   add_outline=False,
                   color_map = 'plasma',
                   ncol = 3,
                   figsize = (6,6)):
    time_key = list(source[list(source.keys())[0]].keys())[0]
    time_cat = adata.obs[time_key].cat.categories.tolist()
    trace_name = list(source.keys())

    aggr_rep = []
    for x in trace_name:
        trace = adata.obs[f'trace_{x}'].values

        rep = adata.obsm[basis][:,:2][trace>0]
        time = np.array(adata.obs[time_key])[trace>0]
        trace = trace[trace>0]
        mean_rep = []
        time_index = []
        for t in time_cat:
            if t in list(time):
                mask = time == t
                mean_rep.append(trace[mask].dot(rep[mask])/trace[mask].sum())
                # mean_rep.append(rep[mask].mean(0))
                time_index.append(t)
        df = pd.DataFrame(np.vstack(mean_rep),columns=['x','y'])
        df[time_key] = pd.Categorical(time_index, time_index)
        df['trace'] = x
        aggr_rep.append(df)
    aggr_rep = pd.concat(aggr_rep)
    aggr_rep['trace'] = pd.Categorical(aggr_rep['trace'].values, trace_name)
    aggr_rep.reset_index(inplace=True)

    if color is None:
        color = time_key
    if not 'trace_label_colors' in adata.uns.keys():
        adata.uns['trace_label_colors'] = [mcolors.to_hex(i) for i in plt.cm.tab10(np.linspace(0,1,len(adata.obs['trace_label'].cat.categories.tolist())))]
    trace_color = dict(zip(adata.obs['trace_label'].cat.categories.tolist(), adata.uns['trace_label_colors']))
    time_color = dict(zip(adata.obs[time_key].cat.categories.tolist(), adata.uns[f'{time_key}_colors']))
    if style == 'merge':
        _, ax = plt.subplots(1,1, figsize = figsize)
        for s in trace_name:
            ax.plot(aggr_rep['x'].values[aggr_rep['trace']==s], 
                    aggr_rep['y'].values[aggr_rep['trace']==s],
                    lw=linewidths,linestyle=linestyle,
                    alpha = line_alpha,
                    c=trace_color[s],label = s,zorder = 2)
        for t in time_cat:
            ax.scatter(aggr_rep['x'].values[aggr_rep[time_key]==t], 
                       aggr_rep['y'].values[aggr_rep[time_key]==t], 
                       alpha = point_alpha,
                       c=time_color[t],s=point_size,
                       edgecolors = edgecolors,
                       linewidth = edgewidth,
                       label = t,zorder=3)

        sc.pl.embedding(adata, basis = basis,color = color, ax = ax, 
                        size = size, add_outline=add_outline,
                        color_map=color_map, show=False)
    
    if style == 'single':
        ncol = min(len(trace_name), ncol)
        if len(trace_name) == ncol:
            nc = ncol
            nr = 1
            pp = list(np.arange(ncol))
            rm_pp = pp.copy()
        elif len(trace_name) > ncol:
            nc = ncol # number of columns
            nr = np.ceil(len(trace_name)/nc).astype(int) # number of rows
            pp = list(itertools.product(np.arange(nr), np.arange(nc)))
            rm_pp = pp.copy()   
        _, axs = plt.subplots(nr,nc, figsize = figsize)
        for i in range(len(trace_name)):
            if nc == nr == 1:
                ax = axs
                rm_pp = None
            elif (nc == 1) | (nr ==1):
                ax = axs[pp[i]]
                rm_pp.pop(0)
            else:
                ax = axs[pp[i][0], pp[i][1]]
                rm_pp.pop(0)

            sub_agg_rep = aggr_rep[aggr_rep['trace']==trace_name[i]]
            ax.plot(sub_agg_rep['x'].values, 
                    sub_agg_rep['y'].values,
                    lw=linewidths,
                    linestyle=linestyle,
                    alpha = line_alpha,
                    c=trace_color[trace_name[i]],
                    # label = trace_name[i],
                    zorder = 2)
            for t in time_cat:
                ax.scatter(sub_agg_rep['x'].values[sub_agg_rep[time_key]==t], 
                           sub_agg_rep['y'].values[sub_agg_rep[time_key]==t], 
                           alpha = point_alpha,
                           c=time_color[t],s=point_size,
                           edgecolors = edgecolors,
                           linewidth = edgewidth,
                           # label = t, 
                           zorder=3)
            if color == 'trace_score':
                sc.pl.embedding(adata, basis = basis,
                                color = f'trace_{trace_name[i]}',title=trace_name[i],
                                size = size,
                                add_outline = add_outline,
                                ax = ax, color_map=color_map, show=False)
            if color == 'trace_label':
                val = np.array(adata.obs['trace_label'])
                val[val != trace_name[i]] = 'other'
                # adata.obs['new_label11'] = 
                sub_adata = adata[val=='other'].concatenate(adata[val==trace_name[i]])
                sub_adata.obs = pd.DataFrame(index=sub_adata.obs.index.values)
                sub_adata.obs['trace_label'] = pd.Categorical(np.concatenate([val[val=='other'], val[val==trace_name[i]]]),
                                                              ['other',trace_name[i]])
                
                sub_adata.uns['trace_label_colors'] = ['#727272', trace_color[trace_name[i]]]
                sc.pl.embedding(sub_adata, basis = basis,
                                color = 'trace_label', 
                                title=trace_name[i], 
                                ax = ax, show=False)                

        # remove unused subplots
        if not rm_pp == None:
            if len(rm_pp) > 0:
                for rmplot in rm_pp:
                    if nc == 1 | nr ==1:  
                        axs[rmplot].set_axis_off()
                    else:
                        axs[rmplot[0], rmplot[1]].set_axis_off()   

def pick_colors(adata, key, colors = None):
    '''
    Create a dict of colors for adata.obs[key]
    '''
    if ~pd.api.types.is_categorical_dtype(adata.obs[key]):
        adata.obs[key] = pd.Categorical(np.array(adata.obs[key]))
    cats = cat(adata.obs, key, return_val = True)
    if colors is not None:
        colrs = dict(zip(cats, colors))
    elif '%s_colors'%key in adata.uns:
        colrs = dict(zip(cats, adata.uns['%s_colors'%key]))
    else:
        colrs = {cats[i]:'C%s'%s for i in range(len(cats))}
        adata.uns['%s_colors'] = [colrs[x] for x in colrs]
        print('Random create colors for %s'%key)
    return(colrs)

def steam_plot(adata, 
               groups, 
               deg=None, 
               y = 'logfoldchanges',
               thr = 0.3, 
               pval_thr = 1e-3,
               size = 4,
               colors = None,
               deg_id='deg',
               vmin = None,
               vmax = None,
               figsize = (8,6)):
    '''
    Stem plot of DEGs
    
    Parameters:
    ----------
    adata : anndata.AnnData.
    groups : groups for comparison.
    deg : DataFrame of DEGs calculated by de.FindMarkers.
    y : key of deg for plotting in y axis.
    thr : threshold of y.
    pval_thr : pvalue threshold.
    size : point size.
    colors : Dict of colors for each condition.
    deg_id : Key of DEGs dataframe in adata.uns.
    vmin : min value of y axis.
    vmax : max value of y axis.
    figsize : figure size.

    ----------

    Return:
    ----------
    Steam plot of deg between groups in all conditions.
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> deg = de.FindMarkers(adata, groupby='leiden', condition = 'dataset')
    >>> de.steam_plot(adata, groups=['cluster0','cluster1'], deg = deg, y = 'logfoldchanges', thr = 0.2, pval_thr = 1e-3, colors=  {'dataset1': '#34649E','dataset2': '#3B44FF'})
    '''

    if deg is None:
        if deg_id in adata.uns:
            deg = adata.uns[deg_id]
        else:
            raise('Cannot find deg result, run de.FindMarkers first.')
            
    condition = factorOrder(deg['condition'])
    fig, ax = plt.subplots(1,ncols=1,sharex = True, sharey=True, figsize = figsize)
    sub_deg1 = deg[deg.group == groups[0]]
    sub_deg2 = deg[deg.group == groups[1]]

    cat = dict(zip(condition, np.arange(len(condition))))
    
    x1 = [cat[i] for i in sub_deg1['condition']]
    x2 = [cat[i] for i in sub_deg2['condition']]

    c1 = np.array(['#4F4F4F'] * len(x1)); c1[(sub_deg1[y]>thr)&(sub_deg1['pvals']<pval_thr)] = '#DD4E4E'
    c2 = np.array(['#4F4F4F'] * len(x2)); c2[(sub_deg2[y]>thr)&(sub_deg2['pvals']<pval_thr)] = '#DD4E4E'
    
    bias1 = np.random.uniform(-0.3, 0.3, len(x1))
    bias2 = np.random.uniform(-0.3, 0.3, len(x2))

    # 背景矩形
    for i in range(len(condition)): 
        high = max(sub_deg1[sub_deg1['condition'] == condition[i]][y])
        low = max(sub_deg2[sub_deg2['condition'] == condition[i]][y])
        
        rect = patches.Rectangle((-0.4+1*i, -low), 0.4*2, low+high, linewidth=1, edgecolor=None, facecolor='#D2D2D2', alpha = 0.3)
        ax.add_patch(rect)


    ax.scatter(x1+bias1, sub_deg1[y], s=size, c=c1)
    ax.scatter(x2+bias2, -sub_deg2[y],s=size,  c=c2)

        # 中央矩形
    for i in range(len(condition)):
        if colors is not None:
            facecolor = colors[condition[i]]
        else:
            facecolor = 'C%s'%i
        rect = patches.Rectangle((-0.5+1*i, -thr), 1, thr*2, linewidth=1, edgecolor='#000000', facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(-0+1*i, 0, condition[i], fontsize=12, color='#000000',horizontalalignment='center', verticalalignment='center')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_ylim(vmin,vmax)

    yticks = ax.get_yticks()
    yticks_positive = [abs(tick) for tick in yticks]
    ax.set_yticks(yticks,yticks_positive)
    ax.set_ylabel(y)
    
    ax.set_xticks(np.arange(len(condition)),condition, rotation=90)


def number_of_deg(adata=None,
                  deg=None,
                  deg_id='deg',
                  color1 = '#77C3A5',
                  color2 = '#FBBA00',
                  vmin = None,
                  vmax = None,
                  share_axis = False,
                  only_target_group=False,
                  swap_axis = False,
                  bbox_to_anchor=(1,0.5,0.5,0.5),
                  figsize = (4,12)):
    '''
    plot number of DEGs between each of group in each conditoin
    
    Parameters:
    ----------
    adata : anndata.AnnData.
    deg : DataFrame of DEGs calculated by de.FindMarkers
    deg_id : Key of DEGs dataframe in adata.uns
    color1 : colors of condition 1
    color2 : colors of condition 2
    share_axis : Share x or y axis
    only_target_group : Only show the number of DEGs of target group
    swap_axis : Swap xaxis and yaxis
    bbox_to_anchor: Position of legend
    figsize : figure size
    ----------

    Return:
    ----------
    DataFrame of number of DEGs
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> deg = de.FindMarkers(adata, groupby='leiden', condition = 'dataset')
    >>> ndeg = de.number_of_deg(adata)
    '''
    if deg is None:
        if deg_id in adata.uns:
            deg = adata.uns[deg_id]
        else:
            raise('Cannot find deg result, run de.FindMarkers first.')
    if 'condition' not in deg.columns.tolist():
        deg['condition'] = ''
    group = factorOrder(deg.group)
    condition = factorOrder(deg.condition)
    if swap_axis:
        sharex = share_axis
        sharey=True
    else:
        sharex = True
        sharey=share_axis
    fig, axs = plt.subplots(nrows=len(group),ncols=1,sharex = sharex, sharey=sharey, figsize = figsize)
    if len(group) == 1:
        axs = [axs]
    dfs = []
    for i in range(len(group)):
        i1 = []; i2 = []
        c1 = colors; c2 = color2
        for x in condition:
            sub_deg = deg[deg.condition == x]
            i1.append((sub_deg.group == group[i]).sum())
            name1 = group[i]
            c1 = color1
            if only_target_group:
                i2.append(0)
                name2 = ''
                name = name1
                if adata is not None:
                    if '%s_colors'%adata.uns['deg_group'] in adata.uns:
                        color_anno = dict(zip(adata.obs[adata.uns['deg_group']].cat.categories.tolist(),adata.uns['%s_colors'%adata.uns['deg_group']]))
                        c1 = color_anno[group[i]]
                        
            else:
                i2.append((sub_deg.group != group[i]).sum())
                if len(factorOrder(sub_deg.group)) == 2:
                    name2 = (set(group) - {group[i]}).pop()
                else:
                    name2 = 'Others'  
                name = '%s_vs_%s'%(name1,name2)
        
        ndeg = pd.DataFrame({name1:i1, name2:i2},index = condition)        
        ndeg['condition'] = condition
        ndeg_return = pd.DataFrame({'numberOfDEG':i1},index = condition)
        ndeg_return['condition'] = condition
        ndeg_return['group'] = name
        dfs.append(ndeg_return)
        if swap_axis:
            axs[i].barh(ndeg['condition'],ndeg[name1], color = c1, label = name1) 
            axs[i].barh(ndeg['condition'],-ndeg[name2], color = c2, label = name2)
            axs[i].set_xlim(vmin,vmax)
            xticks = axs[i].get_xticks()
            xticks_positive = [abs(int(tick)) for tick in xticks]
            axs[i].set_xticks(xticks,xticks_positive)
            axs[i].set_yticks(np.arange(ndeg.shape[0]),ndeg['condition'])
            axs[i].axvline(x=0, color='#000000')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            if i != (len(group)-1):
                axs[i].tick_params(axis='y', which='both', bottom=False, top=False,left=False, labelleft=True)
            else:
                axs[i].set_xlabel('Number of DEGs')
                axs[i].tick_params(axis='y', which='both', bottom=False, top=False,left=False, labelleft=True)
        
        else:
            axs[i].bar(ndeg['condition'],ndeg[name1], color = c1, label = name1) # 蓝色
            axs[i].bar(ndeg['condition'],-ndeg[name2], color = c2, label = name2) # 紫色
            axs[i].set_ylim(-vmin,vmax)
            yticks = axs[i].get_yticks()
            yticks_positive = [abs(int(tick)) for tick in yticks]
            axs[i].set_yticks(yticks,yticks_positive)
            axs[i].set_xticks(np.arange(ndeg.shape[0]),ndeg['condition'], rotation=90)
            axs[i].set_ylabel('Number of DEGs') 
            axs[i].axhline(y=0, color='#000000')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            if i != (len(group)-1):
                axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        axs[i].xaxis.grid(False)
        axs[i].yaxis.grid(False)
        axs[i].legend(bbox_to_anchor=bbox_to_anchor,frameon=False)
    return(pd.concat(dfs))


def enrich_barplot(adata, 
                   enrich=None, 
                   nterms = 5,
                   colorby = 'Cluster', # Cluster
                   sharex = False, 
                   min_lfc=-8,
                   max_lfc=8,
                   cmap = None,
                   hspace=0.5,
                   enrich_id = 'enrich',
                   figsize = (4,8),
                   filename = None):
    '''
    Barplot for enrichment result

    Parameters:
    ----------
    adata: anndata.AnnData.
    enrich: User selected terms for plotting. If None, I will use adata.uns['enrich']
    nterms: Number of terms for plot, sorted by pvalue.
    colorby: Color of bar, can be Cluster or logfoldchanges.
    height: Height of bar.
    sharex: share x axis.
    min_lfc: min values for color map.
    max_lfc: max values for color map.
    cmap: colormap, used when colorby is Cluster
    filename: filename for save.
    '''
    if enrich is None:
        enrich = adata.uns[enrich_id]
    if 'GeneRatio' in enrich.columns.tolist():
        deg = adata.uns['deg']
    if 'NES' in enrich.columns.tolist():
        deg = adata.uns['lfc']
    enrich['-log10(pval)'] = -np.log10(enrich['pvalue'].values.astype(float))
    df = enrich.groupby('Cluster').apply(lambda x: x.nlargest(nterms, '-log10(pval)')).reset_index(drop=True)
    df = df.sort_values(by='-log10(pval)')
    if cmap is None:
        cmap = colors.LinearSegmentedColormap.from_list('bwr', ['#424D9F','#ffffff','#E42A1E'])
    clus = np.unique(df['Cluster'])
    n = len(clus)
    if sharex:
        nx = n
    else:
        nx = n+1
    if colorby is 'Cluster':
        nx = len(clus)
    fig, ax = plt.subplots(nrows=nx,ncols=1,sharex=sharex,figsize = figsize)
    plt.subplots_adjust(hspace=hspace)
    for i in range(n):
        sub_df = df[df.Cluster==clus[i]]
        # if colorby is 'logfoldchanges': 

        color = '#ffffff'
        edgecolor='#000000'
        if colorby is 'Cluster':
            edgecolor=None
            if 'deg_group' in adata.uns:
                groupby = adata.uns['deg_group']
                if ('%s_colors'%adata.uns['deg_group'] in adata.uns) & (len(adata.uns['%s_colors'%adata.uns['deg_group']]) == len(clus)):
                    colr = dict(zip(adata.obs[groupby].cat.categories.tolist(), adata.uns['%s_colors'%groupby]))
                    color = colr[clus[i]]
                else:
                    color = 'C%s'%i
            else:
                color = 'C%s'%i
        ax[i].barh(np.arange(sub_df.shape[0]), 
           sub_df['-log10(pval)'].values, 
           height = 0.8,linewidth=1.5,
           color = color, edgecolor=edgecolor)
        if colorby is 'logfoldchanges':
            sub_df = sub_df[sub_df['geneID']!='']
            lfc = get_lfc(sub_df, deg,min_val=min_lfc,max_val=max_lfc)
            sub_df_with_genes = sub_df[sub_df['geneID']!='']
            for j in range(sub_df.shape[0]):
                name = sub_df['Description'].values[j]
                unit = sub_df['-log10(pval)'].values[j]/len(lfc[name])
                y_h = [unit]*len(lfc[name])
                data_cum = np.array(y_h).cumsum()
                
                for x in range(len(y_h)):
                    starts = data_cum[x] - unit
                    ax[i].barh(j, unit, left=starts, color = cmap(lfc[name].values[x]), height = 0.8, edgecolor='none')
        ax[i].set_yticks(np.arange(sub_df.shape[0]),sub_df['Description'].values)
        ax[i].set_title(clus[i])
        ax[i].yaxis.grid(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        if i == (n-1):
            ax[i].set_xlabel('-log10(pval)')
        # legend
        if colorby is 'logfoldchanges':
            if not sharex:
                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                ax[n].imshow(gradient,aspect='auto',cmap = cmap)
                ax[n].xaxis.grid(False)
                ax[n].yaxis.grid(False)
                ax[n].set_yticks([])
                ax[n].set_xticks([0,256],[min_lfc,max_lfc])
                ax[n].set_position([0.15,0.3, 0.2, 0.02])
                ax[n].set_xlabel('logFC')
    if filename is not None:
        plt.savefig(filename)

def get_lfc(df, deg, normalize=True,max_val=None,min_val=None):
    '''
    Get genes logfoldchage values from deg dataframe.
    '''
    lfc_val = {}
    for i in range(df.shape[0]):
        group = df['Cluster'].values[i]
        genes = df['geneID'].values[i].split('/')
        if isinstance(deg, dict):
            lfc = deg[group]
        else:
            sub_deg = deg[deg['group'] == group]
            lfc = pd.Series(sub_deg['logfoldchanges'].values, index = sub_deg['names'].values)
        lfc_val[df['Description'].values[i]] = lfc[genes].sort_values()
    if normalize:
        if min_val is None:
            mins = min([min(lfc_val[x]) for x in lfc_val])
        else:
            mins = min_val
        if max_val is None:
            maxs = max([max(lfc_val[x]) for x in lfc_val])
        else:
            maxs = max_val
        lfc_val = {x:(lfc_val[x]-mins)/(maxs-mins) for x in lfc_val}
    return(lfc_val)


def plot_lr_link(lr,
                 adata=None, 
                 groupby=None,
                 condition = None, 
                 line_width = 1, 
                 defalut_link_color = '#000000',
                 filename='./italk.pdf'):
    '''
    Circulized plot of ligand-receptor link.

    Parameters:
    ----------
    lr : DataFrame of ligand-receptor results from de.cellchat.
    adata : anndata.AnnData.
    groupby : The group id in adata.obs to calculate ligand-receptor interaction.
    conditoin : Condition to comparison.
    line_width : line width of link.
    defalut_link_color : Used when condition is None, the dafault color of links.
    filename : filename of pdf plot to save.
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> lr = de.cellchat(adata, groupby='celltype', condition = 'condition', n_samples = 500)
    >>> dlr = de.FindDI(lr)
    >>> de.plot_lr_link(dlr ,adata=adata, groupby = 'celltype', condition = 'condition')
    ------
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects.lib.ggplot2 as ggplot2
    if groupby is None:
        if adata is not None:
            if 'cellchat_group' in adata.uns:
                group=robjects.StrVector(factorOrder(adata.obs[adata.uns['cellchat_group']]))
                group_color = get_group_colors(adata, adata.uns['cellchat_group'])
                group_color = robjects.StrVector([group_color[i] for i in group])
        else:
            group=robjects.StrVector(['group'])
            group_color = robjects.StrVector(['#000000'])
    else:
        group=robjects.StrVector(factorOrder(adata.obs[groupby]))
        group_color = get_group_colors(adata, groupby)
        group_color = robjects.StrVector([group_color[i] for i in group])
        
    if condition is None: 
        if adata is not None:
            if 'cellchat_condition' in adata.uns:
                con=robjects.StrVector(factorOrder(adata.obs[adata.uns['cellchat_condition']]))
                conditoin_color = get_group_colors(adata, adata.uns['cellchat_condition'])
                conditoin_color = robjects.StrVector([conditoin_color[i] for i in con])
        lr['condition'] = 'fix'
        con=robjects.StrVector(['fix'])
        conditoin_color = robjects.StrVector([defalut_link_color])
    else:
        con=robjects.StrVector(factorOrder(adata.obs[condition]))
        conditoin_color = get_group_colors(adata, condition)
        conditoin_color = robjects.StrVector([conditoin_color[i] for i in con])
    
    if isinstance(line_width,str):
        lw = robjects.FloatVector(lr[line_width]).r_repr()
    elif isinstance(line_width, int) or isinstance(line_width, float):
        lw = line_width
    else:
        lw = 1
    lr.rename(columns = {'source':'cell_from',
                         'target':'cell_to',
                         'annotation':'comm_type',
                         'condition':'label'}, inplace = True)
    if 'label' not in lr.columns.tolist():
        lr['label'] = 'cellchat'
    r('library(iTALK)')
    with (robjects.default_converter + pandas2ri.converter).context():
        df = robjects.conversion.get_conversion().py2rpy(lr)
    r('df=%s'%df.r_repr())
    r('df$cell_from = as.vector(df$cell_from)')
    r('df$cell_to = as.vector(df$cell_to)')
    r('cell_col = structure(%s,names = %s)'%(group_color.r_repr(), group.r_repr()))
    r('arr_col = structure(%s,names = %s);arr_col = arr_col[df$label]'%(conditoin_color.r_repr(), con.r_repr()))
    r('if(length(cell_col)==1){cell_col=NULL}')
    grdevices = importr('grDevices')
    grdevices.pdf(file='%s'%filename)
    r("LRPlot(df,datatype='mean count',cell_col=cell_col,link.arr.col = arr_col,link.arr.lwd=%s)"%lw)
    grdevices.dev_off()
    print('Save plot as %s'%filename)

def plot_inout(lr, 
               adata=None,
               type = 'count', # prob
               size=300, 
               sizeby=None,
               groupby=None,
               condition=None,
               group_color=None,
               show_grid = False,
               xmin = None,
               xmax = None,
               ymin = None,
               ymax = None,
               legend_pos = (0.2, -0.6),
               figsize=(10,8)):
    """
    Generate scatter plots of incoming vs outgoing ligand-receptor (LR) interaction signals per cell type.
    Parameters
    ----------
    lr : pandas.DataFrame
        DataFrame containing LR interaction data with required columns:
        - 'source': Source cell type (outgoing signal)
        - 'target': Target cell type (incoming signal)
        - 'prob': LR interaction probability/score (used for count/sum calculations)
        - 'condition' (optional): Experimental condition (if stratifying plots)
    adata : anndata.AnnData, optional
        AnnData object containing single-cell metadata (required for cell proportion calculations, 
        group colors, and default groupby/condition values)
    type : str, optional (default='count')
        Metric to calculate incoming/outgoing signals:
        - 'count': Number of LR interactions (count of prob values per cell type)
        - 'prob': Sum of LR interaction probabilities (sum of prob values per cell type)
    size : int, optional (default=300)
        Base size of scatter plot dots (if sizeby=None) or scaling factor (if sizeby='cell_proportion')
    sizeby : str or None, optional (default=None)
        Variable to scale dot size (only supports 'cell_proportion' currently; None = fixed size)
    groupby : str or None, optional (default=None)
        Column name in adata.obs defining cell type groups (matches 'source'/'target' in lr DataFrame).
        If None, attempts to use adata.uns['cellchat_group']; if unavailable, raises a warning.
    condition : str or None, optional (default=None)
        Column name in adata.obs defining experimental conditions (stratifies plots by condition).
        If None, attempts to use adata.uns['cellchat_condition']; if unavailable, uses a single 
        "Income and outcome signals" condition.
    group_color : dict or None, optional (default=None)
        Custom color mapping for cell types (keys = cell type names, values = hex/rgb colors).
        If None, uses colors from get_group_colors(adata, groupby).
    show_grid : bool, optional (default=False)
        Whether to display grid lines on scatter plots for easier readability.
    xmin/xmax : float or None, optional (default=None)
        Minimum/maximum values for the x-axis (outgoing signals); None = auto-scale.
    ymin/ymax : float or None, optional (default=None)
        Minimum/maximum values for the y-axis (incoming signals); None = auto-scale.
    legend_pos : tuple, optional (default=(0.2, -0.6))
        Deprecated/Unused in current implementation (legacy parameter for legend positioning).
    figsize : tuple, optional (default=(10,8))
        Overall figure size (width, height) in inches.

    Returns
    -------
    None
        Generates and displays matplotlib figure; saves no files (figure can be saved externally with plt.savefig()).

    """
    if groupby is None:
        if 'cellchat_group' in adata.uns:
            groupby = adata.uns['cellchat_group']
        else:
            print('Please specify the groupby in adata.obs for ligand-receptor calculation.')
    ratios = {}
    if condition is None:
        if 'cellchat_condition' in adata.uns:
            condition = adata.uns['cellchat_condition']
            for c in factorOrder(lr['condition']):
                sub_obs = adata.obs[adata.obs[condition] == c]
                ratios[c] = sub_obs[groupby].value_counts()/sub_obs[groupby].value_counts().sum()
        else:
            ratios['Income and outcome signals'] = adata.obs[groupby].value_counts()/adata.obs[groupby].value_counts().sum()
            lr = lr.copy()
            lr['condition'] = 'Income and outcome signals'
    else:
        for c in factorOrder(lr['condition']):
            sub_obs = adata.obs[adata.obs[condition] == c]
            ratios[c] = sub_obs[groupby].value_counts()/sub_obs[groupby].value_counts().sum()
    conds = factorOrder(lr['condition'])
    # group 颜色
    group_colors=get_group_colors(adata, groupby)

    plt.rcParams['ytick.left'] = True 
    plt.rcParams['xtick.bottom'] = True  
    plt.rcParams['axes.edgecolor'] = 'black' 
    fig,axs = plt.subplots(2,len(conds),figsize = figsize,sharex = False, sharey = False)
    for i in range(len(conds)):
        c = conds[i]
        sub_lr = lr[lr['condition'] == c]
        if type == 'count':
            ins = sub_lr.groupby(['target'])['prob'].count()
            out = sub_lr.groupby(['source'])['prob'].count()
        if type == 'prob':
            ins = sub_lr.groupby(['target'])['prob'].sum()
            out = sub_lr.groupby(['source'])['prob'].sum()        
        celltype = out.index.values
        df = pd.DataFrame({'outgoing':out[celltype].values,
                           'incomming':ins[celltype].values,
                           'cell_proportion': ratios[c][celltype].values},
                           index = celltype)
        handles = []
        if len(conds) == 1:
            ax_main = axs[i]
            ax_legend = axs[i+1]
        else:
            ax_main = axs[0,i]
            ax_legend = axs[1,i]
        for j in range(len(celltype)):

            if sizeby is None:
                s = size
            else:
                s = df.loc[celltype[j], 'cell_proportion']*size
            plot = ax_main.scatter(df.loc[celltype[j], 'outgoing'], df.loc[celltype[j], 'incomming'], 
                                    s=s, color = group_colors[celltype[j]], label = celltype[j])
            ax_main.text(df.loc[celltype[j], 'outgoing']+(max(df['outgoing'])-min(df['outgoing']))*0.03, 
                          df.loc[celltype[j], 'incomming']+(max(df['outgoing'])-min(df['incomming']))*0.03,
                          abbreviate_word(celltype[j]))
        ax_main.set_xlabel('Outgoing %s'%type)
        ax_main.set_ylabel('Incomming %s'%type)
        ax_main.set_title(c)
        ax_main.grid(show_grid)
        if xmin is not None or xmax is not None:
            ax_main.set_xlim(xmin,xmax)
        if ymin is not None or ymax is not None:
            ax_main.set_ylim(ymin,ymax)
        if sizeby is None:
            fig.delaxes(ax_legend)
        else:
            create_dot_size_legend(df['cell_proportion'], ax_legend,interval=2,
                                   times = size,title = 'Cell proportion')
        plt.subplots_adjust(wspace=0.4)

def plot_interaction(lr,
                     groupby,
                     adata, 
                     alpha=1,
                     title = 'Celltype interaction',
                     filename = 'cellchat_interaction.pdf'):
    """
    Generate alluvial (sankey) plot for ligand-receptor (LR) cell-cell interaction analysis.
    
    This function visualizes the frequency of LR interactions between source and target cell types,
    converting Python/pandas data structures to R objects to leverage the ggalluvial package for
    alluvial plot generation. 

    Parameters
    ----------
    lr : pandas.DataFrame
        DataFrame containing ligand-receptor interaction data with at minimum:
        - 'source': Column with source cell type identifiers
        - 'target': Column with target cell type identifiers
    groupby : str
        Column name in adata.obs defining cell type groups (matches 'source'/'target' in lr DataFrame)
    adata : anndata.AnnData
        AnnData object containing single-cell data with cell type annotations in adata.obs[groupby]
        Used to get cell type order and color palettes consistent with the dataset
    alpha : float, optional (default=1)
        Transparency (alpha) value for stratum (rectangular) elements in the plot (0=transparent, 1=opaque)
    title : str, optional (default='Celltype interaction')
        Title for the alluvial plot
    filename : str, optional (default='cellchat_interaction.pdf')
        Output filename (including .pdf extension) for saving the generated plot

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame with columns:
        - 'source': Source cell type
        - 'target': Target cell type
        - 'freq': Count of LR interactions between source-target pair
        - 'ratio': Percentage of total interactions (formatted as "X.X%")
    """
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects.lib.ggplot2 as ggplot2
    lr['label'] = [lr.source.values[i]+'->'+lr.target.values[i] for i in range(lr.shape[0])]
    df1 = lr.groupby(['source','target'])['label'].count().reset_index()
    df1.rename(columns = {'label':'freq'},inplace = True)
    df1['ratio'] = ['%s%%'%i for i in round(df1['freq']/df1['freq'].sum()*100,1)]
    df2 = pd.DataFrame({groupby:np.concatenate([df1['source'].values, df1['target'].values]),
                        'freq':np.concatenate([df1['freq'].values, df1['freq'].values]),
                        'id':np.concatenate([np.arange(df1.shape[0]),np.arange(df1.shape[0])]),
                        'group':np.concatenate([np.repeat('Source',df1.shape[0]), np.repeat('Target',df1.shape[0])]),
                        'ratio':np.concatenate([np.array(['']*df1.shape[0]), df1['ratio'].values])})
    df2[groupby] = pd.Categorical(np.array(df2[groupby]),factorOrder(adata.obs, groupby))
    df2['group'] = pd.Categorical(np.array(df2['group']),['Source','Target'])
    gr_color = get_group_colors(adata, groupby)
    gr_color=robjects.StrVector([gr_color[i] for i in gr_color])
    with (robjects.default_converter + pandas2ri.converter).context():
        df2 = robjects.conversion.get_conversion().py2rpy(df2)
    ggalluvial = importr('ggalluvial')
    r('library(ggalluvial);library(dplyr);df2=%s;colors=%s'%(df2.r_repr(), gr_color.r_repr()))
    expand=robjects.FloatVector([0.3, 0.1])
    gp = ggplot2.ggplot(df2)
    pp = (gp
        + ggplot2.aes_string(x = 'group', stratum = groupby, alluvium = 'id', y = 'freq', fill = groupby, label = groupby)
        + ggplot2.scale_x_discrete(expand = expand)
        + ggalluvial.geom_flow()
        + ggalluvial.geom_stratum(alpha = alpha)
        + ggplot2.geom_text(ggplot2.aes_string(label = 'ratio'),stat = "flow", size = 3, nudge_x = .25) 
        + ggplot2.geom_text(stat = 'stratum', size = 3)
        + ggplot2.ggtitle(title)
        + ggplot2.theme_minimal() 
        + ggplot2.theme(**{'panel.grid':ggplot2.element_blank(),
                           'axis.text.y':ggplot2.element_blank(),
                           'axis.title.x':ggplot2.element_blank(),
                           'axis.title.y':ggplot2.element_blank()})
        + ggplot2.scale_fill_manual(values=gr_color)
    )
    r("ggsave('%s')"%filename)
    print('Save file <%s>'%filename)
    return(df1)