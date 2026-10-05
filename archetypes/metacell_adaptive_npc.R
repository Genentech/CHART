
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

outdir = paste0("tmp_archetype_outputs/",channel)
indir = paste0("/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/tmp_archetype_outputs/",channel)

dir.create(outdir, recursive = TRUE)

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
                 paste0(outdir,"/npc_elbow_gamma",gamma,"_knn",k.knn,".pdf"),
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

# Aggregate to mean
aggregated <- supercell_GE(t(pca), supercells$membership)

# Write metacell coordinates
write_parquet(as.data.frame(t(as.matrix(aggregated))),
              paste0(outdir,'/supercells-',channel,'_gamma',gamma,
                     '_knn',k.knn,'_npc',n.pc,'.parquet'))
message("Supercell PC coordinates saved to parquet file")
