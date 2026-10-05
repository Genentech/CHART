# Choose how many PCs to keep for a channel, from the elbow of the per-PC
# variance curve. 

load_pca <- function(path) {
  pca <- read_parquet(path)
  cell_id <- paste(pca$Well, pca$Label, sep = "-")
  rownames(pca) <- cell_id
  # The last four columns are Well/Label/Guide/Gene, not PCs.
  pca[,1:(ncol(pca)-4)]
}

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

plot_npc_elbow <- function(sel, path, title) {
  pdf(path, width=6, height=4)
  plot(sel$idx, log(sel$var[sel$idx]), type="b", xlab="PC", ylab="log(var)",
       main=title)
  abline(v = sel$npc, lty = 2, col = "red")
  dev.off()
}

# Entrypoint, skipped when this file is sourced
# -----------------------------
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("Usage: Rscript pick_npc.R <cell_pca_centered.parquet> <npc.txt> [npc]")
  }
  pca_path <- args[[1]]
  out_path <- args[[2]]

  if (length(args) >= 3) {
    npc <- as.numeric(args[[3]])
    if (is.na(npc)) stop("npc given as '", args[[3]], "', which is not a number")
    message("Using the npc given on the command line: ", npc)
  } else {
    library(arrow)
    sel <- pick_npc_by_elbow(load_pca(pca_path), npc_min=10, npc_max=30,
                             use_log=TRUE)
    npc <- sel$npc
    message("Elbow selected n.pc = ", npc, " from ", pca_path)
    plot_npc_elbow(sel, sub("\\.txt$", ".pdf", out_path),
                   paste0("Selected n.pc = ", npc))
  }

  writeLines(as.character(npc), out_path)
  message("Wrote ", out_path)
}
