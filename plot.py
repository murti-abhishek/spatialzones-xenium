# spatialzones/plot.py
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ============================================================
#  Spatial and expression visualization utilities
# ============================================================

def plot_region_gene_spatial(temp, gene: str):
    """
    Plot spatial distribution of regions and gene expression.
    Uses region colors stored in temp.uns['region_colors'].

    Parameters
    ----------
    temp : AnnData
        AnnData object containing spatial data with `region` annotations.
    gene : str
        Gene name to visualize expression for.
    """
    # verify region colors exist
    if 'region_colors' not in temp.uns:
        raise ValueError("Missing temp.uns['region_colors']. Run assign_tumor_regions first.")
    if 'region' not in temp.obs:
        raise ValueError("Missing temp.obs['region']. Run assign_tumor_regions first.")

    sq.pl.spatial_scatter(
        temp,
        library_id="spatial",
        color=["region", gene],
        shape=None,
        size=0.5,
        img=False,
        ncols=2
    )


# ============================================================
#  Regional composition
# ============================================================

def plot_region_composition(temp):
    """
    Compute and plot the composition of cell types within each region.
    Uses stored region colors from temp.uns['region_colors'].
    """
    if 'region' not in temp.obs or 'temp_annotations' not in temp.obs:
        raise ValueError("Missing required obs columns: ['region', 'temp_annotations'].")

    region_order = list(temp.obs['region'].cat.categories)
    region_palette = dict(zip(region_order, temp.uns['region_colors']))

    comp = []
    for r in region_order:
        idx = np.where(temp.obs['region'] == r)[0]
        if len(idx) == 0:
            continue
        types = temp.obs.iloc[idx]['temp_annotations']
        cnt = types.value_counts()
        s = cnt / cnt.sum()
        comp.append(s.rename(r))

    if not comp:
        raise ValueError("No valid regions found in 'temp.obs['region']'.")

    comp_df = pd.concat(comp, axis=1).fillna(0)
    comp_df['Cell_Type'] = comp_df.index.astype(str)
    plot_df = comp_df.melt(id_vars='Cell_Type', var_name='Region', value_name='Fraction')

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x='Cell_Type',
        y='Fraction',
        hue='Region',
        hue_order=region_order,
        palette=region_palette
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Fraction within region")
    plt.xlabel("Cell type")
    plt.title("Regional composition of cell types")
    plt.tight_layout()
    plt.show()


# ============================================================
#  Violin plot for gene by region
# ============================================================

def plot_gene_violin_by_region(temp, gene: str):
    """
    Violin plot of a gene's expression across regions using stored colors.
    """
    if 'region_colors' not in temp.uns:
        raise ValueError("Missing temp.uns['region_colors']. Run assign_tumor_regions first.")

    sc.pl.violin(
        temp,
        keys=gene,
        groupby='region',
        rotation=90,
        stripplot=False
    )


# ============================================================
#  Dotplot for gene by region
# ============================================================

def plot_gene_dotplot_by_region(temp, gene: str):
    """
    Dotplot of a gene's expression across regions.
    """
    sc.pl.dotplot(
        temp,
        var_names=[gene],
        groupby='region',
        cmap='viridis',
        standard_scale='var'
    )


# ============================================================
#  Graph vs Euclidean distance
# ============================================================

def plot_graph_vs_euclidean(temp):
    """
    Scatter plot comparing graph and Euclidean distance to tumor cells, colored by region.
    Adds Spearman correlation.
    """
    if not all(x in temp.obs.columns for x in ['graph_dist_to_tumor', 'euclid_dist_to_tumor']):
        raise ValueError("AnnData missing distance columns. Run assign_tumor_regions first.")

    region_order = list(temp.obs['region'].cat.categories)
    region_palette = dict(zip(region_order, temp.uns['region_colors']))

    rho, p = spearmanr(temp.obs['graph_dist_to_tumor'], temp.obs['euclid_dist_to_tumor'])

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=temp.obs,
        x='graph_dist_to_tumor',
        y='euclid_dist_to_tumor',
        hue='region',
        hue_order=region_order,
        palette=region_palette,
        alpha=0.7,
        s=20
    )
    sns.regplot(
        data=temp.obs,
        x='graph_dist_to_tumor',
        y='euclid_dist_to_tumor',
        scatter=False,
        color='black',
        line_kws={'lw': 1, 'ls': '--', 'alpha': 0.7}
    )
    plt.title("Graph vs Euclidean distance to tumor")
    plt.text(
        0.05, 0.95,
        f"Spearman r = {rho:.2f}\np = {p:.1e}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.tight_layout()
    plt.show()
