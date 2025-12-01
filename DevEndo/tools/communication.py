import scanpy as sc
import anndata
import gc,os,pkgutil
from .core import *
from .plot import *

def cellchat(adata, 
             groupby,
             condition = None,
             species='mouse',
             cells_per_group = 1000,
             type = 'truncatedMean',
             saveDir = 'cellchat',
             saveObj = 'lr.rds'):
    '''
    Random select n_samples cells from each group
    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The group id in adata.obs to calculate ligand-receptor interaction.
    conditoin : Condition to separately calculation.
    species : mouse or human
    cells_per_group : Number of cells to randomly choice in each group.
    type : methods for computing the average gene expression per cell group. Options are "triMean", "truncatedMean", "thresholdedMean", "median".
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> deg = de.cellchat(adata, groupby='leiden', n_samples = 500)
    ------
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    os.makedirs(saveDir, exist_ok = True)
    adata1 = choiceGroupCells(adata, groupby, cells_per_group)
    if condition == None:
        adata1 = anndata.AnnData(adata1.X, obs = adata1.obs.loc[:,[groupby]], var = adata1.var.loc[:,[]])
        lr = run_cellchat(adata1, groupby = groupby, species = species, type = type, saveDir=saveDir,saveObj = groupby+'-'+saveObj)
    else:
        res = []
        adata1 = anndata.AnnData(adata1.X, obs = adata1.obs.loc[:,[groupby, condition]], var = adata1.var.loc[:,[]])
        for c in factorOrder(adata1.obs[condition]):
            sub_adata = adata1[adata1.obs[condition] == c]
            if len(factorOrder(sub_adata.obs[groupby])) > 1:
                print('Run cellchat for %s'%c)
                sub_res = run_cellchat(sub_adata, groupby = groupby, species = species, type = type, saveDir=saveDir, saveObj = groupby+'-'+c+'-'+saveObj)
                sub_res['condition'] = c
                res.append(sub_res)
        lr = pd.concat(res)
        adata.uns['cellchat_condition'] = condition
    adata.uns['cellchat_group'] = groupby
    return(lr)

def run_cellchat(adata, 
                 groupby,
                 species='mouse',
                 type = 'truncatedMean',
                 saveDir = None,
                 saveObj = None):
    '''
    Random select n_samples cells from each group
    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The group id in adata.obs to calculate ligand-receptor interaction.
    species : mouse or human
    cells_per_group : Number of cells to randomly choice in each group.
    type : methods for computing the average gene expression per cell group. Options are "triMean", "truncatedMean", "thresholdedMean", "median".
    ----------
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    import anndata2ri
    r('adata = %s'%anndata2ri.py2rpy(adata).r_repr())
    r('library(Seurat);library(CellChat);CellChatDB=CellChatDB.%s;PPI = PPI.%s'%(species,species),print_r_warnings=False)
    r("CellChatDB.use=subsetDB(CellChatDB, search = c('Secreted Signaling','ECM-Receptor','Cell-Cell Contact'), key = 'annotation')",print_r_warnings=False)
    r("sub_adata=createCellChat(object = assay(adata), meta = as.data.frame(colData(adata)), group.by ='%s')"%groupby,print_r_warnings=False)
    r("sub_adata@DB=CellChatDB.use;sub_adata=subsetData(sub_adata)",print_r_warnings=False)
    r("sub_adata=identifyOverExpressedGenes(sub_adata)",print_r_warnings=False)
    r("sub_adata=identifyOverExpressedInteractions(sub_adata)",print_r_warnings=False)
    r("sub_adata= projectData(sub_adata, PPI)",print_r_warnings=False)
    r("sub_adata=computeCommunProb(sub_adata,type = '%s')"%type,print_r_warnings=False)
    r("sub_adata=filterCommunication(sub_adata, min.cells = 10);sub_adata <- computeCommunProbPathway(sub_adata);sub_adata <- aggregateNet(sub_adata)",print_r_warnings=False)
    if saveObj is not None:
        r("sub_adata@data=matrix();saveRDS(sub_adata, file='%s/%s')"%(saveDir,saveObj))
        print("Save cellchat object to '%s/%s'"%(saveDir,saveObj))
    lr = r("subsetCommunication(sub_adata)",print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
        lr = robjects.conversion.get_conversion().rpy2py(lr)
    r('rm(adata);rm(sub_adata);rm(CellChatDB.use)')
    r('gc()')
    gc.collect()
    return(lr)

def get_lr_from_cellchat_obj(filename):
    '''
    Get ligand-receptor dataframe from cellchat object.
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    r('library(CellChat);adata=loadRDS(filename)',print_r_warnings=False)
    lr = r("subsetCommunication(sub_adata)",print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
        lr = robjects.conversion.get_conversion().rpy2py(lr)
    r('rm(adata);gc()')
    return(lr)

def get_cellchat_interaction(species = 'mouse'):
    '''
    Get interaction database from cellchat.
    --------
    Parameters
    species : mouse or human
    Usage:
    ------
    >>> import devEndo as de
    >>> db = de.get_cellchat_interaction('mouse')
    ------
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    r('library(Seurat);library(CellChat);CellChatDB=CellChatDB.%s;PPI = PPI.%s'%(species,species),print_r_warnings=False)
    db = r("CellChatDB$interaction",print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
        db = robjects.conversion.get_conversion().rpy2py(db)
    return(db)

def get_cellchat_genes(species = 'mouse'):
    '''
    Get genes database from cellchat.
    --------
    Parameters
    species : mouse or human
    Usage:
    ------
    >>> import devEndo as de
    >>> db = de.get_cellchat_genes('mouse')
    ------
    '''
    from rpy2.robjects import r
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    r('library(Seurat);library(CellChat);CellChatDB=CellChatDB.%s;PPI = PPI.%s'%(species,species),print_r_warnings=False)
    db = r("CellChatDB$geneInfo",print_r_warnings=False)
    with (robjects.default_converter + pandas2ri.converter).context():
        db = robjects.conversion.get_conversion().rpy2py(db)
    return(db)

def FindDI(lr, retain_common_lr=False):
    '''
    Identify the specific ligand-receptor for each condition. Only use when condition is not None.
    Parameters:
    ----------
    lr : cellchat results.
    retain_common_lr : retain the lrs enrich for all conditions.
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> lr = de.cellchat(adata, groupby='celltype', condition = 'condition', n_samples = 500)
    >>> lrs = FindDI(res)
    ------
    '''
    if 'condition' not in lr.columns.tolist():
        raise('No condition found in lr columns, please run de.cellchat with condition.')
    dfs = []
    condition = lr['condition'].unique()
    source = lr['source'].unique()
    target = lr['target'].unique()
    for i in source:
        for j in target:
            sub_lr = lr[(lr['source'] == i)&(lr['target'] == j)]
            sub_condition = sub_lr['condition'].unique()
            if len(sub_condition)>1:
                sub_lr = pd.concat([sub_lr[sub_lr['condition']==i] for i in sub_condition])
                sub_lr_raw = sub_lr.copy()
                counts = sub_lr['interaction_name'].value_counts()
                used = counts.index.values[counts==1]
                common = counts.index.values[counts==len(sub_condition)]
                sub_lr = sub_lr[sub_lr['interaction_name'].isin(used)]
                sub_lr['id'] = [sub_lr['source'].values[i]+sub_lr['target'].values[i]+sub_lr['interaction_name'].values[i] for i in range(sub_lr.shape[0])]
                sub_lr.sort_values('prob',ascending = False,inplace = True)
                sub_lr_dedup = sub_lr.drop_duplicates(subset='id', keep='first')
                lr_dict = split(sub_lr_dedup['interaction_name'], sub_lr_dedup['condition'])
                lr_dict = {i:lr_dict[i] for i in list(sub_condition) if i in lr_dict}
                lr_ij = pd.concat([sub_lr[sub_lr['interaction_name'].isin(lr_dict[i])] for i in lr_dict])
                if retain_common_lr:
                    lr_common = sub_lr_raw[sub_lr_raw['interaction_name'].isin(common)]
                    lr_common['id'] = 'common'
                    lr_ij = pd.concat([lr_ij, lr_common])
                dfs.append(lr_ij)
    dfs = pd.concat(dfs)
    dfs.reset_index(inplace = True)
    dfs = dfs.drop('index', axis=1)
    dfs['id2'] = np.where(
        dfs['id'] != 'common',
        dfs['source'] + '->' + dfs['target'] + '(' + dfs['condition'] + ')',
        'common'
    )
    dfs['id'] = [dfs['source'][i]+'->'+dfs['target'][i]+'('+dfs['condition'][i]+')' for i in range(dfs.shape[0])]
    return(dfs)

def plot_lr(lr, 
            x = 'interaction_name_2', 
            y = 'id', 
            title = 'Condition specific interaction',
            sort = True,
            size = 180,
            xtick_colors=None,
            cmap = None, 
            swap_axis = False,
            figsize = (20,8),
            x_margin = 0.01,
            y_margin = 0.04,
            ax=None):
    '''
    Dot plot for ligand-receptor in each condition.
    '''
    if 'id2' not in lr.columns.tolist():
        lr['id2'] = lr['source'].astype(object) + '->' + lr['target'].astype(object) + '(' + lr['condition'].astype(object) + ')'
    if 'id' in lr.columns.tolist():
        common = lr[lr['id2'] == 'common']
        sub_lr = lr[lr['id2'] != 'common']
    else:
        lr['id'] = lr['source'].astype(object) + '->' + lr['target'].astype(object) + '(' + lr['condition'].astype(object) + ')'
        common = pd.DataFrame(columns = lr.columns.tolist())
        sub_lr = lr
    if sort:
        cat = factorOrder(sub_lr['condition'])
        lr = pd.concat([sub_lr[sub_lr['condition'] == i] for i in cat])
        lr = pd.concat([lr, common])
    if ax is None:
        fig, ax1 = plt.subplots(1,1,figsize  = figsize)
    else:
        ax1 = ax
    if swap_axis:
        x0 = y
        y0 = x
    else:
        x0 = x
        y0 = y
    if cmap is None:
        color_map = mymap_reds
    else:
        color_map = cmap
    scatter = ax1.scatter(lr[x0], lr[y0], 
               s = size,c=lr['prob'],
               edgecolors = 'black', linewidths = 1,
               cmap = color_map)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.margins(x=x_margin)
    ax1.margins(y=y_margin)
    ax1.set_title(title)
    if xtick_colors is not None:
        for i, label in enumerate(ax1.get_xticklabels()):
            key = list(xtick_colors.keys())[0]
            x = lr[key].values[i]
            label.set_color(xtick_colors[key][x])
    if swap_axis:
        ax1.invert_yaxis()
    if ax is None:
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Score')
        return(ax1)

def liana(adata, 
             groupby,
             condition = None,
             species='mouse',
             cells_per_group = 1000):
    '''
    Wrap of liana+ cell-cell interaction analysis
    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The group id in adata.obs to calculate ligand-receptor interaction.
    conditoin : Condition to separately calculation.
    species : mouse or human
    cells_per_group : Number of cells to randomly choice in each group.
    type : methods for computing the average gene expression per cell group. Options are "triMean", "truncatedMean", "thresholdedMean", "median".
    ----------
    
    Usage:
    ------
    >>> import devEndo as de
    >>> deg = de.liana(adata, groupby='leiden',species='human', n_samples = 500)
    ------
    '''
    adata1 = choiceGroupCells(adata, groupby, cells_per_group)
    if condition == None:
        adata1 = anndata.AnnData(adata1.X, obs = adata1.obs.loc[:,[groupby]], var = adata1.var.loc[:,[]])
        lr = run_liana(adata1, groupby = groupby, species = species)
    else:
        res = []
        adata1 = anndata.AnnData(adata1.X, obs = adata1.obs.loc[:,[groupby, condition]], var = adata1.var.loc[:,[]])
        for c in factorOrder(adata1.obs[condition]):
            sub_adata = adata1[adata1.obs[condition] == c]
            if len(factorOrder(sub_adata.obs[groupby])) > 1:
                print('Run liana for %s'%c)
                sub_res = run_liana(sub_adata, groupby = groupby, species = species)
                sub_res['condition'] = c
                res.append(sub_res)
        lr = pd.concat(res)
        adata.uns['liana_condition'] = condition
    adata.uns['liana_group'] = groupby
    return(lr)


def run_liana(adata, 
             groupby,
             species='mouse'):
    '''
    Run liana+ cell-cell interaction
    Parameters:
    ----------
    adata : anndata.AnnData.
    groupby : The group id in adata.obs to calculate ligand-receptor interaction.
    species : mouse or human
    ----------
    '''
    if pkgutil.find_loader('liana') is None:
        os.system(f"{sys.executable} -m pip install liana decoupler openpyxl pandas==2.1.0")
    from liana.mt import rank_aggregate
    from liana.method import singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean
    import liana as li
    import decoupler as dc
    human_resource = li.rs.select_resource('consensus')
    map_df = li.rs.get_hcop_orthologs(url='https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/human_mouse_hcop_fifteen_column.txt.gz',
                                      columns=['human_symbol', 'mouse_symbol'],
                                       min_evidence=3
                                       )
    # rename the columns to source and target, respectively for the original organism and the target organism
    map_df = map_df.rename(columns={'human_symbol':'source', 'mouse_symbol':'target'})
    
    # We will then translate
    mouse_resource = li.rs.translate_resource(human_resource,
                                     map_df=map_df,
                                     columns=['ligand', 'receptor'],
                                     replace=True,
                                     one_to_many=1
                                     )
    if species == 'mouse':
        resource = mouse_resource
    if species == 'human':
        resource = human_resource
    methods = [cellchat]
    new_rank_aggregate = li.mt.AggregateClass(li.mt.aggregate_meta, methods=methods)
    new_rank_aggregate(adata= adata, use_raw = False, groupby=groupby, resource = resource)
    res = adata.uns['liana_res']
    res.rename(columns = {'ligand_complex':'ligand','receptor_complex':'receptor','lr_probs':'prob','cellchat_pvals':'pval'},inplace = True)
    res['interaction_name_2'] = res['ligand'] + ' - '+res['receptor']
    interaction1 = de.get_cellchat_interaction()
    interaction = de.split(interaction1['ligand'],interaction1['annotation'])
    annotation = np.array(['Others']*res.shape[0],dtype = '<U20')
    for i in interaction:
        annotation[res['ligand'].isin([interaction[i]])] = i
    res['annotation'] = annotation
    res['database'] = 'liana'
    res.reset_index(inplace=True,drop=True)
    return(res)