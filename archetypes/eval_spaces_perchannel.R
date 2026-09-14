#!/usr/bin/env Rscript

# Script to evaluate metacell partitions for a given channel
# Usage: Rscript eval_spaces_perchannel.R <channel> [use_pw]
# Example: Rscript eval_spaces_perchannel.R "TOM20" FALSE

suppressPackageStartupMessages({
  library(quadprog)
  library(FNN)
  library(arrow)
})

# -------------------------
# Parse command line arguments
# -------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript eval_spaces_perchannel.R <channel> [use_pw]\n  channel: channel name (e.g., 'TOM20')\n  use_pw: TRUE/FALSE (default: FALSE)")
}

channel <- as.character(args[1])
use_pw <- if (length(args) >= 2) as.logical(args[2]) else FALSE

message("Starting evaluation for channel: ", channel)
message("use_pw: ", use_pw)

# -------------------------
# Parse params from filename (pw or non-pw)
# -------------------------
parse_gkp <- function(path) {
  m <- regexec("gamma([0-9.]+)_knn([0-9.]+)_npc([0-9.]+)(?:_pw)?\\.rds$", basename(path))
  mm <- regmatches(basename(path), m)[[1]]
  if (length(mm) != 4) stop("Could not parse gamma/knn/npc from: ", path)
  list(gamma = as.numeric(mm[2]), knn = as.numeric(mm[3]), npc = as.integer(mm[4]))
}

# -------------------------
# Metacell aggregation
# -------------------------
metacell_means <- function(scores, membership, P) {
  stopifnot(nrow(scores) == length(membership))
  membership <- as.integer(membership)

  levels_g <- sort(unique(membership))
  gf <- factor(membership, levels = levels_g)

  sums <- rowsum(scores[, 1:P, drop = FALSE], gf, reorder = FALSE)
  n_cells <- as.integer(tabulate(gf))
  means <- sweep(as.matrix(sums), 1, pmax(n_cells, 1), "/")

  list(M = means, n_cells = n_cells)
}

# -------------------------
# K-agnostic metacell QC metrics
# -------------------------
metacell_size_summary <- function(n_cells) {
  data.frame(
    n_metacells = length(n_cells),
    mean_n_cells = mean(n_cells),
    median_n_cells = median(n_cells),
    frac_lt5  = mean(n_cells < 5),
    frac_lt10 = mean(n_cells < 10),
    stringsAsFactors = FALSE
  )
}

within_dispersion_summary <- function(scores_full, membership, Pfull = 20) {
  membership <- as.integer(membership)
  stopifnot(nrow(scores_full) == length(membership))

  levels_g <- sort(unique(membership))
  gf <- factor(membership, levels = levels_g)
  n_cells <- as.integer(tabulate(gf))

  X <- as.matrix(scores_full[, 1:Pfull, drop = FALSE])

  sums <- rowsum(X, gf, reorder = FALSE)
  mu_i <- sweep(as.matrix(sums), 1, pmax(n_cells, 1), "/")

  # Per-metacell dispersion: E||x||^2 - ||mu||^2  (always >= 0 up to numerical error)
  x2 <- rowSums(X^2)
  sum_x2 <- as.numeric(rowsum(x2, gf, reorder = FALSE))
  disp_i <- (sum_x2 / pmax(n_cells, 1)) - rowSums(mu_i^2)
  disp_i <- pmax(disp_i, 0)

  # Coherence summary: weighted average within variance per coordinate (stable, never negative)
  within_var_per_coord <- sum(n_cells * disp_i) / (sum(n_cells) * Pfull)

  data.frame(
    within_var_per_coord = within_var_per_coord,
    disp_median = median(disp_i),
    disp_p90 = as.numeric(quantile(disp_i, 0.90)),
    disp_p95 = as.numeric(quantile(disp_i, 0.95)),
    stringsAsFactors = FALSE
  )
}

neighbor_separation_summary <- function(scores_full, membership, Pfull = 20, eps = 1e-8, nn_k = 2) {
  membership <- as.integer(membership)
  levels_g <- sort(unique(membership))
  gf <- factor(membership, levels = levels_g)
  n_cells <- as.integer(tabulate(gf))

  X <- as.matrix(scores_full[, 1:Pfull, drop = FALSE])

  sums <- rowsum(X, gf, reorder = FALSE)
  mu_i <- sweep(as.matrix(sums), 1, pmax(n_cells, 1), "/")

  # intra RMS radius from disp_i
  x2 <- rowSums(X^2)
  sum_x2 <- as.numeric(rowsum(x2, gf, reorder = FALSE))
  disp_i <- (sum_x2 / pmax(n_cells, 1)) - rowSums(mu_i^2)
  intra <- sqrt(pmax(disp_i, 0))

  # nearest-neighbor distance among centroids
  nn <- FNN::get.knn(mu_i, k = nn_k)$nn.dist[, nn_k]
  sep_ratio <- nn / (intra + eps)

  data.frame(
    sep_median = median(sep_ratio),
    sep_p10 = as.numeric(quantile(sep_ratio, 0.10)),
    sep_p25 = as.numeric(quantile(sep_ratio, 0.25)),
    stringsAsFactors = FALSE
  )
}

# -------------------------
# K-agnostic driver: partition evaluation ONLY
# -------------------------
evaluate_partitions <- function(scores_full, supercell_files, Pfull = 20) {
  out <- list()

  for (sc_path in supercell_files) {
    meta <- parse_gkp(sc_path)
    gamma <- meta$gamma; knn <- meta$knn; npc <- meta$npc

    sc <- readRDS(sc_path)
    membership <- sc$membership
    if (is.null(membership)) stop("No membership in: ", sc_path)

    # metacell sizes from membership (doesn't require aggregation)
    n_cells <- as.integer(tabulate(factor(as.integer(membership), levels = sort(unique(as.integer(membership))))))

    sz   <- metacell_size_summary(n_cells)
    disp <- within_dispersion_summary(scores_full, membership, Pfull = Pfull)
    sep  <- neighbor_separation_summary(scores_full, membership, Pfull = Pfull)

    out[[basename(sc_path)]] <- cbind(
      data.frame(gamma = gamma, knn = knn, npc = npc, Pfull = Pfull, stringsAsFactors = FALSE),
      sz, disp, sep
    )

    cat("QC done:", gamma, knn, npc, "\n")
  }

  do.call(rbind, out)
}

# =========================
# OPTIONAL: AA-dependent evaluation layer (keep separate)
# =========================
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
  solve(crossprod(W) + diag(ridge, ncol(W)), crossprod(W, M_full))
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

full_space_recovery_oneK <- function(scores_full, membership, aa_res, npc, K,
                                     Pfull, size_weight_n0, trim_frac, ridge = 1e-8) {
  kstr <- as.character(K)
  rK <- aa_res$results[[kstr]]
  if (is.null(rK$Z_ref)) return(NULL)

  # Metacell means
  agg_npc  <- metacell_means(scores_full, membership, P = npc)
  agg_full <- metacell_means(scores_full, membership, P = Pfull)

  E <- aa_res$eval_indices
  M_npc_eval  <- agg_npc$M[E, , drop = FALSE]
  M_full_eval <- agg_full$M[E, , drop = FALSE]
  n_eval_cells <- agg_npc$n_cells[E]

  M_npc_eval_s <- scale(M_npc_eval)
  W <- project_simplex_qp(M_npc_eval_s, rK$Z_ref)

  Z_full <- fit_Z_full(W, M_full_eval, ridge = ridge)
  Mhat_full <- W %*% Z_full

  w_size <- pmin(1, n_eval_cells / size_weight_n0)

  c(
    n_eval_metacells = length(E),
    rmse_full_unweighted   = rmse_per_coord(M_full_eval, Mhat_full, NULL, 0),
    rmse_full_sizeweighted = rmse_per_coord(M_full_eval, Mhat_full, w_size, 0),
    rmse_full_trimmed      = rmse_per_coord(M_full_eval, Mhat_full, NULL, trim_frac)
  )
}

evaluate_partitions_with_AA <- function(qc_tbl, scores_full, supercell_files, aa_dir,
                                        K = 11, Pfull = 20,
                                        size_weight_n0 = 50, trim_frac = 0.1,
                                        use_pw = FALSE, channel = NULL) {
  # derive AA filename template from use_pw
  aa_template <- if (use_pw) "archetype_robust_runs_gamma%s_knn%s_npc%s_pw.rds"
  else       "archetype_robust_runs_gamma%s_knn%s_npc%s.rds"

  out <- qc_tbl

  # add AA columns
  out$n_eval_metacells <- NA_real_
  out$rmse_full_unweighted <- NA_real_
  out$rmse_full_sizeweighted <- NA_real_
  out$rmse_full_trimmed <- NA_real_
  out$K <- K

  # map sc_path -> row in qc_tbl by (gamma,knn,npc)
  for (sc_path in supercell_files) {
    meta <- parse_gkp(sc_path)
    gamma <- meta$gamma; knn <- meta$knn; npc <- meta$npc

    sc <- readRDS(sc_path)
    membership <- sc$membership
    if (is.null(membership)) next

    aa_path <- file.path(aa_dir, sprintf(aa_template, gamma, knn, npc))
    if (!file.exists(aa_path)) next
    aa_res <- readRDS(aa_path)

    rec <- full_space_recovery_oneK(scores_full, membership, aa_res, npc = npc, K = K,
                                    Pfull = Pfull, size_weight_n0 = size_weight_n0,
                                    trim_frac = trim_frac)

    if (is.null(rec)) next

    i <- which(out$gamma == gamma & out$knn == knn & out$npc == npc)
    if (length(i) == 1) {
      out[i, names(rec)] <- as.numeric(rec)
    }
    cat("AA eval done:", gamma, knn, npc, "\n")
  }

  out
}

# -------------------------
# Main execution
# -------------------------
sc_dir <- file.path("/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/tmp_archetype_outputs", channel)
aa_dir <- sc_dir

# Check if directory exists
if (!dir.exists(sc_dir)) {
  stop("Directory not found: ", sc_dir)
}

# supercell files selection
if (use_pw) {
  supercell_files <- Sys.glob(file.path(sc_dir, sprintf("supercells-%s_gamma*_knn*_npc*_pw.rds", channel)))
} else {
  supercell_files <- Sys.glob(file.path(sc_dir, sprintf("supercells-%s_gamma*_knn*_npc*.rds", channel)))
  supercell_files <- supercell_files[!grepl("_pw\\.rds$", supercell_files)]
}

if (length(supercell_files) == 0) {
  stop("No supercell files found for channel: ", channel)
}

message("Found ", length(supercell_files), " supercell files")

# load scores once (same as you already do)
scores_full <- if (use_pw) {
  pca <- readRDS(file.path(sc_dir, "cell_pca_centered_pw.rds"))
  pca$scores
} else {
  df <- read_parquet(file.path(sc_dir, "cell_pca_centered.parquet"))
  drop_cols <- intersect(colnames(df), c("Well", "Label", "Guide", "Gene"))
  if (length(drop_cols) > 0) df <- df[, !(colnames(df) %in% drop_cols), drop = FALSE]
  as.matrix(df)
}

message("Loaded scores with dimensions: ", nrow(scores_full), " x ", ncol(scores_full))

# 1) K-agnostic partition QC
message("Running K-agnostic partition evaluation...")
qc_tbl <- evaluate_partitions(scores_full, supercell_files, Pfull = ncol(scores_full))

# Save QC results
out_file <- if (use_pw) {
  file.path(sc_dir, "space_evals_pw.csv")
} else {
  file.path(sc_dir, "space_evals.csv")
}

write.csv(qc_tbl, out_file, row.names = FALSE)
message("Saved evaluation results to: ", out_file)

# 2) Optional AA-dependent layer (adds K-specific recovery metrics)
# Uncomment if you want to run AA-dependent evaluation
# message("Running AA-dependent evaluation...")
# tbl <- evaluate_partitions_with_AA(qc_tbl, scores_full, supercell_files, aa_dir,
#                                    K = 11, Pfull = ncol(scores_full),
#                                    size_weight_n0 = 50, trim_frac = 0.1,
#                                    use_pw = use_pw)

message("Evaluation complete for channel: ", channel)
