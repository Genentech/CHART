
# Read arguments from command line
args <- commandArgs(trailingOnly = TRUE)
gamma <- as.numeric(args[1])
k.knn <- as.numeric(args[2])
channel <- as.character(args[[3]])

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
pca <- read_parquet(paste0(indir,"/cell_pca_centered.parquet"))
cell_id <- paste(pca$Well, pca$Label, sep = "-")
rownames(pca) <- cell_id
pca <- pca[,1:(ncol(pca)-4)]
message("PCA data loading complete")


pick_npc_by_elbow <- function(scores, npc_min = 10, npc_max = 60, use_log = TRUE) {
  X <- as.matrix(scores)
  v <- apply(X, 2, var, na.rm = TRUE)
  v[!is.finite(v)] <- 0
  if (sum(v) <= 0) stop("Total variance is zero; are these PCs whitened?")

  npc_max <- min(npc_max, length(v))
  idx <- 1:npc_max

  y <- v[idx]
  if (use_log) y <- log(pmax(y, 1e-12))

  # Points (x, y)
  x <- idx

  # Line through endpoints: (x1,y1) -> (x2,y2)
  x1 <- x[1]; y1 <- y[1]
  x2 <- x[length(x)]; y2 <- y[length(y)]

  # Distance from each point to the line (in 2D)
  # |(y2-y1)x - (x2-x1)y + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
  num <- abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
  den <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  d <- num / (den + 1e-12)

  # Elbow = max distance to chord
  k_elbow <- which.max(d)

  # Clamp
  npc <- max(npc_min, min(k_elbow, npc_max))

  list(
    npc = npc,
    elbow_raw = k_elbow,
    var = v,
    idx = idx,
    score = d
  )
}

# Adaptively select npc for this channel
sel <- pick_npc_by_elbow(pca, npc_min=10, npc_max=30, use_log=TRUE)
n.pc <- sel$npc
message(paste0("Adaptively selected n.pc = ", n.pc, " for channel ", channel))

# Save diagnostic plot
pdf(paste0(outdir,"/npc_elbow_gamma",gamma,"_knn",k.knn,".pdf"), width=6, height=4)
plot(sel$idx, log(sel$var[sel$idx]), type="b", xlab="PC", ylab="log(var)",
     main=paste0(channel, ": Selected n.pc = ", n.pc))
abline(v = sel$npc, lty = 2, col = "red")
dev.off()


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
