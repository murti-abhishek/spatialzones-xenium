# from spatialzones.assign import assign_tumor_regions
# from spatialzones.plot import plot_region_gene_spatial, plot_gene_violin_by_region

# 1) assign for a subset of cores (this returns a temp AnnData)
temp = assign_tumor_regions(
    adata,
    tma_cores=['HB_6_1'],
    tumor_type='Tumor Hepatoblast',
    use_absolute_layers=True,
    inside_hop_val=1,
    interface_hop_val=3
)

# # 2) quick QC plots
plot_region_gene_spatial(temp, 'AHSG')
plot_gene_violin_by_region(temp, 'AHSG')
