
# Read arguments from command line
args <- commandArgs(trailingOnly = TRUE)
gamma <- as.numeric(args[1])
k.knn <- as.numeric(args[2])
channel <- as.character(args[[3]])
# Optional.  Left out, the number of PCs is chosen from the elbow of the
# variance curve, as it always was; a workflow that has to name this
# script's output files before running it passes the number in instead.
n.pc <- if (length(args) >= 4) as.numeric(args[[4]]) else NA
if (length(args) >= 4 && is.na(n.pc)) {
  stop("npc given as '", args[[4]], "', which is not a number")
}

script_dir <- dirname(sub("^--file=", "",
                          grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(script_dir, "pick_npc.R"))

# Every step of this stage reads and writes inside one directory per
# channel.  'chart archetypes' sets it; by hand, export it once.
indir = Sys.getenv("CHART_ARCHETYPE_DIR")
if (indir == "") {
  stop("Set CHART_ARCHETYPE_DIR to the directory holding this channel's ",
       "cell_pca_centered.parquet, or run this through 'chart archetypes'")
}
outdir = indir

# Plots and tables nothing downstream reads go to the reports root when
# 'chart archetypes' names one, and beside the data otherwise.
reportdir = Sys.getenv("CHART_ARCHETYPE_REPORT_DIR")
if (reportdir == "") reportdir = outdir

dir.create(outdir, recursive = TRUE)
dir.create(reportdir, recursive = TRUE, showWarnings = FALSE)

# Load existing env
# if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
# renv::load(project = "/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/")
library(arrow)
library(SuperCell)
library(archetypes)

# Load cell-level PCA space of given channel
pca <- load_pca(paste0(indir,"/cell_pca_centered.parquet"))
message("PCA data loading complete")


# Adaptively select npc for this channel, unless it was given
if (is.na(n.pc)) {
  sel <- pick_npc_by_elbow(pca, npc_min=10, npc_max=30, use_log=TRUE)
  n.pc <- sel$npc
  message(paste0("Adaptively selected n.pc = ", n.pc, " for channel ", channel))

  # Save diagnostic plot
  plot_npc_elbow(sel,
                 paste0(reportdir,"/npc_elbow_gamma",gamma,"_knn",k.knn,".pdf"),
                 paste0(channel, ": Selected n.pc = ", n.pc))
} else {
  message(paste0("Using the n.pc given on the command line: ", n.pc,
                 " for channel ", channel))
}


# Compute metacells ------------------------------------------------------------
supercells <- SCimplify_from_embedding(
  X = pca, # PCA embedding
  k.knn = k.knn, # number of nearest neighbors to build kNN network
  gamma = gamma, # graining level
  n.pc = n.pc
)
message("Supercell computation complete")

# supercells <- readRDS(paste0(outdir,"/supercells-",channel,
#                              "_gamma",gamma,"_knn",k.knn,'_npc',n.pc,".rds"))
# message("Supercell loading complete")

# Save metacell object
saveRDS(supercells, paste0(outdir,"/supercells-",channel,
                           "_gamma",gamma,"_knn",k.knn,'_npc',n.pc,".rds"))
message("Supercell RDS object saved")

# Aggregate to mean.  Only the components the metacells were built from,
# because the file is named after that count and the archetypes step
# writes the same name from the same columns; a wider file here would
# make what the name means depend on which step wrote it last.
aggregated <- supercell_GE(t(pca[, 1:n.pc]), supercells$membership)

# Write metacell coordinates
write_parquet(as.data.frame(t(as.matrix(aggregated))),
              paste0(outdir,'/supercells-',channel,'_gamma',gamma,
                     '_knn',k.knn,'_npc',n.pc,'.parquet'))
message("Supercell PC coordinates saved to parquet file")
