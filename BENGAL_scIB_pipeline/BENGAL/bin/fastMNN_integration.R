# /usr/bin/env R

# © EMBL-European Bioinformatics Institute, 2023
# Yuyao Song <ysong@ebi.ac.uk>

library(optparse)
library(Seurat)
# library(SeuratWrappers)

option_list <- list(
  make_option(c("-i", "--input_rds"),
    type = "character", default = NULL,
    help = "Path to input preprocessed rds file"
  ),
  make_option(c("-o", "--out_rds"),
    type = "character", default = NULL,
    help = "Output fastMNN from Seurat wrappers integrated rds file"
  ),
  make_option(c("-p", "--out_UMAP"),
    type = "character", default = NULL,
    help = "Output UMAP after fastMNN integration"
  ),
  make_option(c("-b", "--batch_key"),
    type = "character", default = NULL,
    help = "Batch key identifier to integrate"
  ),
  make_option(c("-s", "--species_key"),
    type = "character", default = NULL,
    help = "Species key identifier"
  ),
  make_option(c("-c", "--cluster_key"),
    type = "character", default = NULL,
    help = "Cluster key for UMAP plotting"
  )
)

# parse input
opt <- parse_args(OptionParser(option_list = option_list))

input_rds <- opt$input_rds
out_rds <- opt$out_rds
out_UMAP <- opt$out_UMAP
batch_key <- opt$batch_key
species_key <- opt$species_key
cluster_key <- opt$cluster_key

## create Seurat object via rds

# Convert(input_h5ad, dest = "rds", overwrite = TRUE)
# input_rds <- gsub("h5ad", "rds", input_h5ad)
obj <- readRDS(input_rds)

# obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj)


RunFastMNN <- function(
  object.list,
  assay = NULL,
  features = 2000,
  reduction.name = "mnn",
  reduction.key = "mnn_",
  reconstructed.assay = "mnn.reconstructed",
  verbose = TRUE,
  ...
) {

  if (length(x = object.list) < 2) {
    stop("'object.list' must contain multiple Seurat objects for integration",
         call. = FALSE)
  }
  assay <- assay %||% DefaultAssay(object = object.list[[1]])
  for (i in 1:length(x = object.list)) {
    DefaultAssay(object = object.list[[i]]) <- assay
  }
  if (is.numeric(x = features)) {
    if (verbose) {
      message(paste("Computing", features, "integration features"))
    }
    features <- SelectIntegrationFeatures(
      object.list = object.list,
      nfeatures = features,
      assay = rep(assay, length(x = object.list))
    )
  }
  objects.sce <- lapply(
    X = object.list,
    FUN = function(x, f) {
      return(as.SingleCellExperiment(x = subset(x = x, features = f)))
    },
    f = features
  )
  integrated <- merge(
    x = object.list[[1]],
    y = object.list[2:length(x = object.list)]
  )
  out <- do.call(
    what = batchelor::fastMNN,
    args = c(
      objects.sce,
      list(...)
    )
  )
  rownames(x = SingleCellExperiment::reducedDim(x = out)) <- colnames(x = integrated)
  colnames(x = SingleCellExperiment::reducedDim(x = out)) <- paste0(reduction.key, 1:ncol(x = SingleCellExperiment::reducedDim(x = out)))
  integrated[[reduction.name]] <- CreateDimReducObject(
    embeddings = SingleCellExperiment::reducedDim(x = out),
    loadings = as.matrix(SingleCellExperiment::rowData(x = out)),
    assay = DefaultAssay(object = integrated),
    key = reduction.key
  )
  # Add reconstructed matrix (gene x cell)
  integrated[[reconstructed.assay]] <- CreateAssayObject(
    data = as(object = SummarizedExperiment::assay(x = out), Class = "sparseMatrix"),
  )
  # Add variable features
  VariableFeatures(object = integrated[[reconstructed.assay]]) <- features
  Tool(object = integrated) <- S4Vectors::metadata(x = out)
  integrated <- LogSeuratCommand(object = integrated)
  return(integrated)
}

## run fastMNN
obj <- RunFastMNN(object.list = SplitObject(obj, split.by = batch_key))
obj <- RunUMAP(obj, reduction = "mnn", dims = 1:30, n_neighbors = 15L,  min_dist = 0.3)
obj <- FindNeighbors(obj, reduction = "mnn", dims = 1:30)
obj <- FindClusters(obj, resolution = 0.4)

# have to convert all factor to character, or when later converting to h5ad, the factors will be numbers
i <- sapply(obj@meta.data, is.factor)
obj@meta.data[i] <- lapply(obj@meta.data[i], as.character)

saveRDS(obj,
  file= out_rds,
)


pdf(out_UMAP, height = 6, width = 10)
DimPlot(obj, reduction = "umap", group.by = species_key, shuffle = TRUE, label = TRUE)
DimPlot(obj, reduction = "umap", group.by = batch_key, shuffle = TRUE, label = TRUE)
DimPlot(obj, reduction = "umap", group.by = cluster_key, shuffle = TRUE, label = TRUE)

dev.off()
