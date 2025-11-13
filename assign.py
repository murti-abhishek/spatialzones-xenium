# spatialzones/assign.py

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

def assign_tumor_regions(
    adata,
    tma_cores,
    tumor_type: str,
    use_absolute_layers: bool = True,
    inside_hop_val: int = 1,
    interface_hop_val: int = 3,
    connectivity_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    annotation_key: str = "temp_annotations",
    core_key: str = "tma_core",
):
    """
    Assigns 'inside' / 'interface' / 'outside' regions based on graph distance to tumor cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial connectivity and cell annotations.
    tma_cores : str or list of str
        TMA core(s) to process.
    tumor_type : str
        Annotation label identifying tumor cells in `adata.obs[annotation_key]`.
    use_absolute_layers : bool, default=True
        Retained for backward compatibility. Only absolute graph distance is supported.
    inside_hop_val : int, default=1
        Maximum graph distance (in hops) from tumor cells defining the 'inside' region.
    interface_hop_val : int, default=2
        Maximum graph distance from tumor cells defining the 'interface' region.
    connectivity_key : str, default='spatial_connectivities'
        Key in `adata.obsp` containing adjacency or connectivity matrix.
    spatial_key : str, default='spatial'
        Key in `adata.obsm` containing 2D spatial coordinates.
    annotation_key : str, default='temp_annotations'
        Key in `adata.obs` containing cell type annotations.
    core_key : str, default='tma_core'
        Key in `adata.obs` specifying TMA core IDs.

    Returns
    -------
    AnnData
        Subsetted AnnData (for specified cores) with:
        - obs['graph_dist_to_tumor'] : Graph-based distance to nearest tumor cell
        - obs['euclid_dist_to_tumor'] : Euclidean distance to nearest tumor cell
        - obs['region'] : Ordered categorical ('inside', 'interface', 'outside')
        - uns['region_params'] : Metadata about assignment parameters
        - uns['region_colors'] : List of colors for plotting

    Notes
    -----
    - Requires `.obsp[connectivity_key]` to be a symmetric adjacency matrix.
    - Raises ValueError if no tumor cells are found in the subset.
    """
    # Ensure tma_cores is a list
    if isinstance(tma_cores, str):
        tma_cores = [tma_cores]

    # Subset to relevant cores
    temp = adata[adata.obs[core_key].isin(tma_cores)].copy()

    # Sanity checks
    for key, attr in [(connectivity_key, "obsp"), (spatial_key, "obsm"), (annotation_key, "obs")]:
        if key not in getattr(temp, attr):
            raise KeyError(f"Expected '{key}' in adata.{attr}")

    # Prepare adjacency
    A = temp.obsp[connectivity_key]
    if hasattr(A, "toarray"):  # sparse matrix
        A_data = A
    else:
        A_data = np.array(A > 0, dtype=float)

    coords = temp.obsm[spatial_key]

    # Identify tumor cells
    tumor_mask = temp.obs[annotation_key] == tumor_type
    tumor_idx = np.where(tumor_mask)[0]
    if tumor_idx.size == 0:
        raise ValueError(f"No tumor cells of type '{tumor_type}' found in selected cores: {tma_cores}")

    # Compute graph distance (Dijkstra)
    dist_from_tumor = dijkstra(A_data, directed=False, indices=tumor_idx).min(axis=0)
    temp.obs["graph_dist_to_tumor"] = dist_from_tumor

    # Compute Euclidean distance
    kdt = cKDTree(coords[tumor_idx])
    euclid_dists = kdt.query(coords)[0]
    temp.obs["euclid_dist_to_tumor"] = euclid_dists

    # Assign regions
    temp.obs["region"] = "outside"
    temp.obs.loc[temp.obs["graph_dist_to_tumor"] <= inside_hop_val, "region"] = "inside"
    temp.obs.loc[
        (temp.obs["graph_dist_to_tumor"] > inside_hop_val)
        & (temp.obs["graph_dist_to_tumor"] <= interface_hop_val),
        "region",
    ] = "interface"

    region_order = ["inside", "interface", "outside"]
    temp.obs["region"] = pd.Categorical(temp.obs["region"], categories=region_order, ordered=True)

    # Colors
    REGION_COLORS = {
        "inside": "#2ca02c",    # green
        "interface": "#ff7f0e", # orange
        "outside": "#1f77b4",   # blue
    }
    temp.uns["region_colors"] = [REGION_COLORS[r] for r in region_order]

    # Record parameters
    temp.uns["region_params"] = {
        "use_absolute_layers": use_absolute_layers,
        "inside_hop_val": inside_hop_val,
        "interface_hop_val": interface_hop_val,
        "connectivity_key": connectivity_key,
        "spatial_key": spatial_key,
        "annotation_key": annotation_key,
        "core_key": core_key,
    }

    return temp
