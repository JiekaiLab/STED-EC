import itertools
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from .comparison import nonzero
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import cm
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import rc_context
mpl.rcParams['pdf.fonttype']  = 42


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


def create_node_id(ts):
    """
    为轨迹数据创建唯一标识符与标签的映射关系
    
    参数:
    - ts: 包含轨迹数据的对象，需包含trajectory_ds属性(AnnData格式)
    
    功能:
    1. 基于day_field和trajectory_label生成唯一名称
    2. 去除重复项并建立ID到名称/标签的映射
    """
    ts.trajectory_ds.obs['name'] = ts.trajectory_ds.obs[ts.day_field].astype(str) + '_' +ts.trajectory_ds.obs[ts.trajectory_label].astype(str)
    name = ts.trajectory_ds.obs.drop_duplicates(['name'])
    ts.nid = np.arange(name.shape[0])
    ts.name2id = dict(zip(name['name'].values,ts.nid))
    ts.id2label = dict(zip(ts.nid, name[ts.trajectory_label].values))
    ts.id2day = dict(zip(ts.nid, name[ts.day_field].values))
    day_val = ts.adata.obs[ts.day_field].cat.categories.tolist()
    day2pos = dict(zip(day_val, np.arange(len(day_val))))
    ts.id2pos = dict(zip(ts.nid,[day2pos[ts.id2day[i]] for i in ts.nid]))
create_node_id(ts)

def create_sankey_data(ts):
    """
    构建桑基图（Sankey Diagram）所需的数据流数据，
    处理轨迹配对关系，生成包含源、目标、流量值及对应ID的结构化数据

    参数:
    ----
    ts : 包含轨迹数据的对象
        需具备以下属性/结构：
        - couples: 存储轨迹配对关系的字典，键为配对元组(p0, p1)，值为DataFrame（含层级索引）
        - name2id: 字典，映射细胞类型名称到唯一ID（用于桑基图节点编码）
        - sankey_data: 函数执行后，将结果存储在此属性中（DataFrame类型）
    """
    df = []
    for p in ts.couples:
        pair_score = ts.couples[p].stack().reset_index().drop(columns = 'level_0')
        pair_score.rename(columns = dict(zip(pair_score.columns.tolist(),['source','target','value'])),inplace=True)
        pair_score['source'] = str(p[0]) + '_' + pair_score['source']
        pair_score['target'] = str(p[1]) + '_' + pair_score['target']
        # 去掉被过滤的细胞类型
        pair_score = pair_score[pair_score['source'].isin(list(ts.name2id.keys()))]
        pair_score = pair_score[pair_score['target'].isin(list(ts.name2id.keys()))]
        pair_score['source_id'] = [ts.name2id[i] for i in pair_score['source']]
        pair_score['target_id'] = [ts.name2id[i] for i in pair_score['target']]
        df.append(pair_score)
    ts.sankey_data = pd.concat(df)
create_sankey_data(ts)


import plotly.graph_objects as go
thr = np.percentile(ts.sankey_data['value'], 0)
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",  # 使用snap布局使节点按层排列
    node=dict(
        pad=15,
        thickness=10,
        line=dict(color="black", width=0.5),
        label=[ts.id2label[i] for i in ts.nid],
        color=[id2color[i] for i in ts.nid],
        # 为节点添加自定义位置信息（基于层级）
        x=[ts.id2pos[i] for i in ts.nid],  # 三个层级: 0, 0.5, 1
        y=None  # 垂直位置微调
    ),
    link=dict(
        source=ts.sankey_data['source_id'],
        target=ts.sankey_data['target_id'],
        value=ts.sankey_data['value'].values[ts.sankey_data['value']>thr],
        color='#696969',
        hovertemplate='%{source.label} → %{target.label}<br>值: %{value}<extra></extra>'
    )
)])

# 设置图表标题和布局
fig.update_layout(
    title_text="Trajectory score",
    font_size=12,
    width=2000,
    height=600,
    margin=dict(l=50, r=50, t=80, b=50)
)

# 显示图表
fig.show()