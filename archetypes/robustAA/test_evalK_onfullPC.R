suppressPackageStartupMessages({
  library(quadprog)
  library(ggplot2)
  library(arrow)
  library(dplyr)
})

# =============================
# Command line arguments
# =============================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript test_evalK_onfullPC.R <channel> [use_pw]\n",
       "  channel: Channel name (e.g., DAPI1, Golgin97, Fibrillarin)\n",
       "  use_pw:  Optional, TRUE or FALSE (default: FALSE)")
}

channel <- args[1]

# =============================
# Toggle: process only _pw or only non-_pw
# =============================
use_pw <- if (length(args) >= 2) as.logical(args[2]) else FALSE

suffix <- if (use_pw) "_pw" else ""
aa_template <- if (use_pw) {
  "archetype_robust_runs_gamma%s_knn%s_npc%s_pw.rds"
} else {
  "archetype_robust_runs_gamma%s_knn%s_npc%s.rds"
}

# ---------- utilities ----------
parse_gkp <- function(path) {
  # Accept both, so parsing never fails:
  #   ..._npc<*>_pw.rds   OR   ..._npc<*>.rds
  m <- regexec("gamma([0-9.]+)_knn([0-9.]+)_npc([0-9.]+)(?:_pw)?\\.rds$", basename(path))
  mm <- regmatches(basename(path), m)[[1]]
  if (length(mm) != 4) stop("Could not parse gamma/knn/npc from: ", path)
  list(gamma = as.numeric(mm[2]), knn = as.numeric(mm[3]), npc = as.integer(mm[4]))
}

metacell_means <- function(scores, membership, P) {
  membership <- as.integer(membership)
  stopifnot(nrow(scores) == length(membership))
  
  levels_g <- sort(unique(membership))
  gf <- factor(membership, levels = levels_g)
  
  sums <- rowsum(scores[, 1:P, drop = FALSE], gf, reorder = FALSE)
  n_cells <- as.integer(tabulate(gf))
  means <- sweep(as.matrix(sums), 1, pmax(n_cells, 1), "/")
  
  list(M = means, n_cells = n_cells)
}

project_simplex_qp <- function(X, Z) {
  n <- nrow(X); K <- nrow(Z)
  Dmat <- 2 * (Z %*% t(Z)) + diag(1e-8, K)
  Amat <- cbind(rep(1, K), diag(K))
  bvec <- c(1, rep(0, K))
  
  W <- matrix(NA_real_, n, K)
  for (i in seq_len(n)) {
    dvec <- 2 * (Z %*% X[i, ])
    sol <- quadprog::solve.QP(Dmat = Dmat, dvec = dvec, Amat = Amat, bvec = bvec, meq = 1)
    W[i, ] <- sol$solution
  }
  W
}

fit_Z_full <- function(W, M_full, ridge = 1e-8) {
  K <- ncol(W)
  A <- crossprod(W) + diag(ridge, K)
  B <- crossprod(W, M_full)
  solve(A, B)  # K x Pfull
}

rmse_per_coord <- function(M, Mhat, weights = NULL, trim_frac = 0) {
  R <- M - Mhat
  se <- rowSums(R^2)
  keep <- rep(TRUE, length(se))
  
  if (trim_frac > 0) {
    cutoff <- as.numeric(quantile(se, 1 - trim_frac))
    keep <- se <= cutoff
    se <- se[keep]
    if (!is.null(weights)) weights <- weights[keep]
  }
  
  d <- ncol(M)
  if (is.null(weights)) {
    sqrt(sum(se) / (length(se) * d))
  } else {
    w <- as.numeric(weights)
    w <- w / mean(w)
    sqrt(sum(w * se) / (sum(w) * d))
  }
}

# ---------- compute full-space recovery curve for one config ----------
full_recovery_curve_one <- function(scores_full, membership, aa_res,
                                    Pfull = 20,
                                    K_grid = NULL,
                                    size_weight_n0 = 20,
                                    trim_frac = 0.02,
                                    ridge = 1e-8) {
  # infer npc from stored Z_ref for any K that has it
  k_any <- names(aa_res$results)[1]
  if (is.null(aa_res$results[[k_any]]$Z_ref)) stop("AA res missing Z_ref; cannot infer npc.")
  npc <- ncol(aa_res$results[[k_any]]$Z_ref)
  
  if (is.null(K_grid)) K_grid <- sort(as.integer(names(aa_res$results)))
  
  agg_npc  <- metacell_means(scores_full, membership, P = npc)
  agg_full <- metacell_means(scores_full, membership, P = Pfull)
  
  E <- aa_res$eval_indices
  M_npc_eval  <- agg_npc$M[E, , drop = FALSE]
  M_full_eval <- agg_full$M[E, , drop = FALSE]
  n_eval_cells <- agg_npc$n_cells[E]
  
  # Match AA convention: scale metacell npc-space coords
  M_npc_eval_s <- scale(M_npc_eval)
  
  w_size <- pmin(1, n_eval_cells / size_weight_n0)
  
  out <- data.frame(
    K = K_grid,
    rmse_full_unweighted = NA_real_,
    rmse_full_sizeweighted = NA_real_,
    rmse_full_trimmed = NA_real_,
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(K_grid)) {
    k <- as.character(K_grid[i])
    rK <- aa_res$results[[k]]
    if (is.null(rK) || is.null(rK$Z_ref)) next
    
    Z_npc <- rK$Z_ref
    W <- project_simplex_qp(M_npc_eval_s, Z_npc)
    
    Z_full <- fit_Z_full(W, M_full_eval, ridge = ridge)
    Mhat_full <- W %*% Z_full
    
    out$rmse_full_unweighted[i]   <- rmse_per_coord(M_full_eval, Mhat_full, weights = NULL, trim_frac = 0)
    out$rmse_full_sizeweighted[i] <- rmse_per_coord(M_full_eval, Mhat_full, weights = w_size, trim_frac = 0)
    out$rmse_full_trimmed[i]      <- rmse_per_coord(M_full_eval, Mhat_full, weights = NULL, trim_frac = trim_frac)
  }
  
  out
}

# ---------- driver across configs ----------
full_recovery_curves_all <- function(scores_full, supercell_files, aa_dir,
                                     Pfull = 20, K_grid = NULL,
                                     size_weight_n0 = 20, trim_frac = 0.02,
                                     aa_template = "archetype_robust_runs_gamma%s_knn%s_npc%s_pw.rds") {
  all <- list()
  
  for (sc_path in supercell_files) {
    meta <- parse_gkp(sc_path)
    gamma <- meta$gamma; knn <- meta$knn; npc <- meta$npc

    sc <- readRDS(sc_path)
    membership <- sc$membership
    if (is.null(membership)) stop("No membership in: ", sc_path)

    aa_path <- file.path(aa_dir, sprintf(aa_template, gamma, knn, npc))
    if (!file.exists(aa_path)) {
      cat("Warning: Skipping config (gamma=", gamma, ", knn=", knn, ", npc=", npc,
          ") - AA file not found: ", aa_path, "\n", sep="")
      next
    }
    aa_res <- readRDS(aa_path)
    
    curve <- full_recovery_curve_one(
      scores_full = scores_full,
      membership = membership,
      aa_res = aa_res,
      Pfull = Pfull,
      K_grid = K_grid,
      size_weight_n0 = size_weight_n0,
      trim_frac = trim_frac
    )
    
    curve$gamma <- gamma
    curve$knn <- knn
    curve$npc <- npc
    curve$config <- paste0("g", gamma, "_k", knn, "_p", npc, suffix)
    
    all[[basename(sc_path)]] <- curve
    cat("Done curves:", curve$config[1], "\n")
  }
  
  do.call(rbind, all)
}

# ---------- main execution ----------
sc_dir <- file.path("/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/tmp_archetype_outputs", channel)
aa_dir <- sc_dir

cat("Processing channel:", channel, "\n")
cat("use_pw:", use_pw, "\n")

# Select ONLY the desired supercell files
if (use_pw) {
  supercell_files <- Sys.glob(file.path(sc_dir, sprintf("supercells-%s_gamma*_knn*_npc*_pw.rds", channel)))
} else {
  supercell_files <- Sys.glob(file.path(sc_dir, sprintf("supercells-%s_gamma*_knn*_npc*.rds", channel)))
  supercell_files <- supercell_files[!grepl("_pw\\.rds$", supercell_files)]
}

if (length(supercell_files) == 0) {
  stop("No supercell files found for channel: ", channel)
}
cat("Found", length(supercell_files), "supercell files\n")

# Read cell PC scores once
if (use_pw) {
  pca <- readRDS(file.path(sc_dir, "cell_pca_centered_pw.rds"))
  scores_full <- pca$scores
} else {
  scores_full <- read_parquet(file.path(sc_dir, "cell_pca_centered.parquet"))
}

# If your parquet includes metadata columns, drop them safely
drop_cols <- intersect(colnames(scores_full), c("Well", "Label", "Guide", "Gene"))
if (length(drop_cols) > 0) {
  # keep rownames if Well/Label exist
  if (all(c("Well", "Label") %in% colnames(scores_full))) {
    rownames(scores_full) <- paste(scores_full$Well, scores_full$Label, sep = "_")
  }
  scores_full <- scores_full[, !(colnames(scores_full) %in% drop_cols), drop = FALSE]
}
scores_full <- as.matrix(scores_full)

df_full <- full_recovery_curves_all(scores_full, supercell_files, aa_dir,
                                    Pfull = ncol(scores_full), K_grid = 6:20,
                                    size_weight_n0 = 50, trim_frac = 0.1,
                                    aa_template = aa_template)

# -------- Save results --------
# Save the full recovery curves data
out_csv <- file.path(sc_dir, sprintf("evalK_recovery_curves%s.csv", suffix))
write.csv(df_full, out_csv, row.names = FALSE)
cat("Saved results to:", out_csv, "\n")

# -------- Normalized overlay plot (excess over min RMSE per config) --------
ycol <- "rmse_full_trimmed"  # or rmse_full_sizeweighted / rmse_full_unweighted

dfN <- df_full %>%
  group_by(config) %>%
  mutate(
    rmse_abs = .data[[ycol]],
    rmse_min = min(rmse_abs, na.rm = TRUE),
    rmse_rel_excess_to_min = (rmse_abs - rmse_min) / rmse_min
  ) %>%
  ungroup()

p <- ggplot(dfN, aes(x = K, y = rmse_rel_excess_to_min, color = config, group = config)) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.5) +
  labs(x = "K", y = "(RMSE - min) / min",
       title = paste0("Normalized to min RMSE (", ycol, ") - ", channel)) +
  theme_bw() +
  theme(legend.position = "bottom")

# Save plot
out_plot <- file.path(sc_dir, sprintf("evalK_recovery_plot%s.png", suffix))
ggsave(out_plot, plot = p, width = 10, height = 7, dpi = 300)
cat("Saved plot to:", out_plot, "\n")

