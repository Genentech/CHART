suppressPackageStartupMessages({
  library(quadprog)
  library(ggplot2)
  library(arrow)
  library(dplyr)
})

# elbow_index: the K recovery curves bend for the same reason the PC
# variance curve does, so they are read the same way.
script_dir <- dirname(sub("^--file=", "",
                          grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(script_dir, "..", "pick_npc.R"))

# =============================
# Command line arguments
# =============================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript test_evalK_onfullPC.R <channel> [use_pw] [gamma] [knn] [k]\n",
       "  channel: Channel name (e.g., DAPI1, Golgin97, Fibrillarin)\n",
       "  use_pw:  Optional, TRUE or FALSE (default: FALSE)\n",
       "  gamma, knn: which config's curve K is chosen from (default 30 10)\n",
       "  k:       Optional, the K to use instead of the knee of that curve")
}

channel <- args[1]

# Which of the swept configs carries on to the per-cell step, and whether
# its K was decided already.  The defaults are the pair the published run
# used for every channel, for running this by hand; 'chart archetypes'
# passes whichever pair the config names.
sel_gamma <- if (length(args) >= 3) args[3] else "30"
sel_knn <- if (length(args) >= 4) args[4] else "10"
sel_k <- if (length(args) >= 5) as.integer(args[5]) else NA_integer_
if (length(args) >= 5 && is.na(sel_k)) {
  stop("k given as '", args[5], "', which is not a whole number")
}

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
    # Carried through from the archetype runs so that whatever rules out a
    # K is visible beside the error it was ruled out against.
    drift_p90 = NA_real_,
    support_min_ref = NA_real_,
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

    if (!is.null(rK$drift_p90)) out$drift_p90[i] <- rK$drift_p90
    if (!is.null(rK$support_min_ref)) out$support_min_ref[i] <- rK$support_min_ref
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

# ---------- choose K from one config's curve ----------
# The recovery error falls with K and has no minimum worth aiming at: the
# metacells it is measured on are drawn from the same pool the archetypes
# were fitted to, so more archetypes always reconstruct them better.  The
# knee is the answer instead, taken over the K values whose archetypes are
# worth having.  An archetype that no metacell leans on, or one that moves
# about between independent runs, is not a real vertex of the data.
pick_k_by_knee <- function(curve, drift_tolerance = 2) {
  usable <- curve[!is.na(curve$rmse_full_trimmed), , drop = FALSE]
  usable <- usable[order(usable$K), , drop = FALSE]
  if (nrow(usable) < 3) {
    stop("Only ", nrow(usable), " K values have a recovery error, too few ",
         "to find a knee in; check the archetype runs completed")
  }

  usable$vetoed <- FALSE
  usable$veto_reason <- ""

  unused <- !is.na(usable$support_min_ref) & usable$support_min_ref == 0
  usable$vetoed[unused] <- TRUE
  usable$veto_reason[unused] <- "an archetype no metacell leans on"

  # Drift is a distance in the scaled PC space, so it has no absolute
  # scale to compare against; it is judged against the other K values.
  if (any(!is.na(usable$drift_p90))) {
    allowed <- drift_tolerance * median(usable$drift_p90, na.rm = TRUE)
    unstable <- (!is.na(usable$drift_p90) & usable$drift_p90 > allowed
                 & !usable$vetoed)
    usable$vetoed[unstable] <- TRUE
    usable$veto_reason[unstable] <- sprintf(
      "archetypes move %.3g between runs, past the %.3g allowed here",
      usable$drift_p90[unstable], allowed)
  }

  keep <- usable[!usable$vetoed, , drop = FALSE]
  if (nrow(keep) < 3) {
    stop("Only ", nrow(keep), " of ", nrow(usable), " K values survive the ",
         "drift and support checks, too few to find a knee in; the recovery ",
         "curves table says what ruled each one out")
  }

  # Both axes onto [0, 1], so the bend does not depend on the units the
  # error happens to be in.
  unit <- function(z) if (diff(range(z)) > 0) (z - min(z)) / diff(range(z)) else z * 0
  bend <- elbow_index(unit(keep$K), unit(keep$rmse_full_trimmed))

  list(k = keep$K[bend$at], table = usable)
}

# ---------- main execution ----------
# Every step of this stage reads and writes inside one directory per
# channel.  'chart archetypes' sets it; by hand, export it once.
sc_dir <- Sys.getenv("CHART_ARCHETYPE_DIR")
if (sc_dir == "") {
  stop("Set CHART_ARCHETYPE_DIR to this channel's directory, or run this ",
       "through 'chart archetypes'")
}
aa_dir <- sc_dir

# The recovery curves and their plot are read to choose K by hand, so
# they go to the reports root when there is one.
report_dir <- Sys.getenv("CHART_ARCHETYPE_REPORT_DIR")
if (report_dir == "") report_dir <- sc_dir
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

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
# -------- Choose K for the config that carries on --------
chosen <- df_full[df_full$gamma == as.numeric(sel_gamma)
                  & df_full$knn == as.numeric(sel_knn), , drop = FALSE]
if (!nrow(chosen)) {
  stop("No recovery curve for gamma=", sel_gamma, " knn=", sel_knn,
       "; the configs with curves are: ",
       paste(unique(df_full$config), collapse = ", "))
}

if (is.na(sel_k)) {
  picked <- pick_k_by_knee(chosen)
  sel_k <- picked$k
  df_full <- merge(df_full, picked$table[, c("config", "K", "vetoed", "veto_reason")],
                   by = c("config", "K"), all.x = TRUE)

  # Loudly, because the knee has never been checked against a curve someone
  # read by hand: the only ones from before this pipeline came from a solver
  # that kept landing in local optima, so they could not settle it.
  message("\n", strrep("=", 72))
  message("K WAS CHOSEN FOR YOU.  The knee of the recovery curve puts it at ",
          sel_k, " for ", channel, ",")
  message("at gamma ", sel_gamma, " knn ", sel_knn, ".  Check that against the ",
          "curve before trusting it:")
  message("  ", file.path(report_dir, sprintf("evalK_recovery_plot%s.png", suffix)))
  message("  ", file.path(report_dir, sprintf("evalK_recovery_curves%s.csv", suffix)))
  message("If you disagree, set archetypes.k for this channel and run this ",
          "step again.")
  message(strrep("=", 72), "\n")
} else {
  cat("Using the K given on the command line:", sel_k, "\n")
}

k_file <- file.path(sc_dir, sprintf("k%s.txt", suffix))
writeLines(as.character(sel_k), k_file)
cat("Wrote", k_file, "\n")

out_csv <- file.path(report_dir, sprintf("evalK_recovery_curves%s.csv", suffix))
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
out_plot <- file.path(report_dir, sprintf("evalK_recovery_plot%s.png", suffix))
ggsave(out_plot, plot = p, width = 10, height = 7, dpi = 300)
cat("Saved plot to:", out_plot, "\n")

