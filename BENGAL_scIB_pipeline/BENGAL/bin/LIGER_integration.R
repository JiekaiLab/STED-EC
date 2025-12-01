# /usr/bin/env R

# © EMBL-European Bioinformatics Institute, 2023
# Yuyao Song <ysong@ebi.ac.uk>

library(optparse)
library(rliger)
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

# rliger function
RunOptimizeALS <- function(
  object,
  k,
  assay = NULL,
  split.by = 'orig.ident',
  lambda = 5,
  thresh = 1e-6,
  max.iters = 30,
  reduction.name = 'iNMF_raw',
  reduction.key = 'riNMF_',
  nrep = 1,
  H.init = NULL,
  W.init = NULL,
  V.init = NULL,
  rand.seed = 1,
  print.obj = FALSE,
  ...
) {
  assay <- assay %||% DefaultAssay(object = object)
  if (IsMatrixEmpty(x = GetAssayData(object = object, slot = 'scale.data'))) {
    stop("Data is unscaled, splease scale before running", call. = FALSE)
  }
  if (is.character(x = split.by) && length(x = split.by) == 1) {
    split.by <- object[[split.by]]
  }
  split.cells <- split(x = colnames(x = object), f = split.by)
  scale.data <- lapply(
    X = split.cells,
    FUN = function(x) {
      return(t(x = GetAssayData(
        object = object,
        slot = 'scale.data',
        assay = assay
      )[, x]))
    }
  )
  # scale.data <- sapply(X = scale.data, FUN = t, simplify = FALSE)
  out <- rliger::optimizeALS(
    object = scale.data,
    k = k,
    lambda = lambda,
    thresh = thresh,
    max.iters = max.iters,
    nrep = nrep,
    H.init = H.init,
    W.init = W.init,
    V.init = V.init,
    rand.seed = rand.seed,
    print.obj = print.obj
  )
  colnames(x = out$W) <- VariableFeatures(object = object)
  object[[reduction.name]] <- CreateDimReducObject(
    embeddings = do.call(what = 'rbind', args = out$H),
    loadings = t(x = out$W),
    assay = assay,
    key = reduction.key
  )
  Tool(object = object) <- sapply(
    X = out$V,
    FUN = function(x) {
      colnames(x = x) <- VariableFeatures(object = object)
      rownames(x = x) <- colnames(x = object[[reduction.name]])
      return(t(x = x))
    },
    simplify = FALSE
  )
  object <- LogSeuratCommand(object = object)
  return(object)
}

RunSNF <- function(
  object,
  split.by = 'orig.ident',
  reduction = 'iNMF_raw',
  dims.use = NULL,
  dist.use = 'CR',
  center = FALSE,
  knn_k = 20,
  k2 = 500,
  small.clust.thresh = knn_k,
  ...
) {
  .Deprecated(
    new = 'RunQuantileNorm',
    msg = paste(
      "This is a deprecated function. Call 'RunQuantileNorm' instead."
    )
  )
}


RunQuantileAlignSNF <- function(
  object,
  split.by = 'orig.ident',
  reduction = 'iNMF_raw',
  reduction.name = 'iNMF',
  reduction.key = 'iNMF_',
  recalc.snf = FALSE,
  ref_dataset = NULL,
  prune.thresh = 0.2,
  min_cells = 2,
  quantiles = 50,
  nstart = 10,
  resolution = 1,
  center = FALSE,
  id.number = NULL,
  print.mod = FALSE,
  print.align.summary = FALSE,
  ...
) {
  message(paste(
      "This is a deprecated function. Calling 'RunQuantileNorm' instead.",
      "Note that not all parameters can be passed to 'RunQuantileNorm'.",
      "It's suggested to run 'louvainCluster' subsequently as well."
    ))

  .Deprecated(
    new = 'RunQuantileNorm',
    msg = paste(
      "This is a deprecated function. Calling 'quantile_norm' instead.",
      "Note that not all parameters can be passed to 'quantile_norm'.",
      "It's suggested to run 'louvainCluster' subsequently as well."
    )
  )

  return(RunQuantileNorm(object,
    split.by = split.by,
    reduction = reduction,
    reduction.name = reduction.name,
    reduction.key = reduction.key,
    quantiles = quantiles,
    ref_dataset = NULL,
    min_cells = 20,
    knn_k = 20,
    dims.use = NULL,
    do.center = FALSE,
    max_sample = 1000,
    eps = 0.9,
    refine.knn = TRUE,
    ...
  ))
}

RunQuantileNorm <- function(
  object,
  split.by = 'orig.ident',
  reduction = 'iNMF_raw',
  reduction.name = 'iNMF',
  reduction.key = 'iNMF_',
  quantiles = 50,
  ref_dataset = NULL,
  min_cells = 20,
  knn_k = 20,
  dims.use = NULL,
  do.center = FALSE,
  max_sample = 1000,
  eps = 0.9,
  refine.knn = TRUE,
  ...
) {
  embeddings <- sapply(
    X = SplitObject(object = object, split.by = split.by),
    FUN = function(x) {
      return(Embeddings(object = x[[reduction]]))
    },
    simplify = FALSE,
    USE.NAMES = TRUE
  )
  if (is.null(x = ref_dataset)) {
    num.samples <- vapply(
      X = embeddings,
      FUN = nrow,
      FUN.VALUE = integer(length = 1L)
    )
    ref_dataset <- names(x = embeddings)[which.max(x = num.samples)]
  } else if (is.numeric(x = ref_dataset)) {
    ref_dataset <- names(x = embeddings)[ref_dataset]
  }
  if (is.character(x = ref_dataset) && !ref_dataset %in% names(x = embeddings)) {
    stop("Cannot find reference dataset '", ref_dataset, "' in the split", call. = FALSE)
  }
  out <- rliger::quantile_norm(
    object = embeddings,
    quantiles = quantiles,
    ref_dataset = ref_dataset,
    min_cells = min_cells,
    knn_k = knn_k,
    dims.use = dims.use,
    do.center = do.center,
    max_sample = max_sample,
    eps = eps,
    refine.knn = refine.knn,
    ...
  )
  object[[reduction.name]] <- CreateDimReducObject(
    embeddings = out$H.norm,
    assay = DefaultAssay(object = object[[reduction]]),
    key = reduction.key
  )
  out <- as.data.frame(x = out[names(x = out) != 'H.norm'])
  object[[colnames(x = out)]] <- out
  Idents(object = object) <- 'clusters'
  object <- LogSeuratCommand(object = object)
  return(object)
}


## create Seurat object via rds

# Convert(input_h5ad, dest = "rds", overwrite = TRUE)
# input_rds <- gsub("h5ad", "rds", input_h5ad)
obj <- readRDS(input_rds)

obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj, split.by = batch_key, do.center = FALSE)

# LIGER
obj <- RunOptimizeALS(obj, k = 30, lambda = 5, split.by = batch_key)
obj <- RunQuantileNorm(obj, split.by = batch_key)

pbmcLiger <- runIntegration(obj, k = 30, lambda = 5,)

obj <- FindNeighbors(obj, reduction = "iNMF", dims = 1:30)
obj <- FindClusters(obj, resolution = 0.4)
# Dimensional reduction and plotting
obj <- RunUMAP(obj, dims = 1:ncol(obj[["iNMF"]]), reduction = "iNMF", n_neighbors = 15L,  min_dist = 0.3)


# have to convert all factor to character, or when later converting to h5ad, the factors will be numbers
i <- sapply(obj@meta.data, is.factor)
obj@meta.data[i] <- lapply(obj@meta.data[i], as.character)


saveRDS(obj,
  file = out_rds
)

# iNMF embedding will be in .obsm['iNMF']

pdf(out_UMAP, height = 6, width = 10)
DimPlot(obj, reduction = "umap", group.by = species_key, shuffle = TRUE, label = TRUE)
DimPlot(obj, reduction = "umap", group.by = batch_key, shuffle = TRUE, label = TRUE)
DimPlot(obj, reduction = "umap", group.by = cluster_key, shuffle = TRUE, label = TRUE)

dev.off()
