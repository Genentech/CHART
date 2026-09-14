# Read an embedding file, run an arbitrary preprocessing operation,
# then pass the resulting matrix to run_one_space() (NOT run_one_space_file()).
#
# Usage:
#   Rscript aa_select_one_space_customprep.R --in spaceA.rds --out AA_select_spaceA.rds
#
# Optional args:
#   --kmin 6 --kmax 20 --runs 8 --nrep 10 --fit_size 12000 --eval_size 8000 --C_size 12000
#   --knn_k 30 --alpha 1.5 --seed 1 --tau 0.3 --fracCfit 0.7 --fracCeval 0.5

# if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
# renv::load(project = "/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/")

library(archetypes)
library(FNN)
library(quadprog)
library(clue)
library(SuperCell)
library(arrow)

# -----------------------------
# Simple argument parsing
# -----------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) return(list())
  out <- list(); i <- 1
  while (i <= length(args)) {
    key <- args[i]
    if (!startsWith(key, "--")) stop("Bad arg: ", key)
    key <- sub("^--", "", key)
    if (i == length(args)) stop("Missing value for --", key)
    out[[key]] <- args[i + 1]
    i <- i + 2
  }
  out
}
as_int <- function(x, default) if (is.null(x)) default else as.integer(x)
as_num <- function(x, default) if (is.null(x)) default else as.numeric(x)
as_chr <- function(x, default) if (is.null(x)) default else as.character(x)

# -----------------------------
# Core utilities
# -----------------------------
scale_embed <- function(X) scale(X)

knn_rarity <- function(X, k = 30) FNN::get.knn(X, k = k)$nn.dist[, k]

fps_indices <- function(X, m = 12000, seed = 1L) {
  set.seed(seed)
  n <- nrow(X); m <- min(m, n)
  sel <- integer(m)
  sel[1] <- sample.int(n, 1)
  
  dmin <- rowSums((X - matrix(X[sel[1], ], n, ncol(X), byrow = TRUE))^2)
  for (t in 2:m) {
    j <- which.max(dmin)
    sel[t] <- j
    dj <- rowSums((X - matrix(X[j, ], n, ncol(X), byrow = TRUE))^2)
    dmin <- pmin(dmin, dj)
  }
  sel
}

sample_mix <- function(n_total, C_idx, rarity, m, frac_C, alpha, seed) {
  set.seed(seed)
  mC <- min(length(C_idx), floor(m * frac_C))
  mR <- m - mC
  
  idxC <- sample(C_idx, mC, replace = FALSE)
  
  p <- rarity^alpha
  p[!is.finite(p)] <- 0
  p <- p / sum(p)
  idxR <- sample.int(n_total, mR, replace = FALSE, prob = p)
  
  unique(c(idxC, idxR))
}

fit_best_AA <- function(X_fit, K, nrep = 10, seed = 1L) {
  set.seed(seed)
  bm <- bestModel(stepArchetypes(X_fit, k = K, nrep = nrep, verbose = FALSE))
  list(Z = parameters(bm), rss_train = rss(bm))
}

project_simplex_qp <- function(X, Z) {
  n <- nrow(X); K <- nrow(Z)
  Dmat <- 2 * (Z %*% t(Z)) + diag(1e-8, K)
  
  Amat <- cbind(rep(1, K), diag(K))   # sum(w)=1 and w>=0
  bvec <- c(1, rep(0, K))
  
  W <- matrix(NA_real_, n, K)
  for (i in seq_len(n)) {
    dvec <- 2 * (Z %*% X[i, ])
    sol <- quadprog::solve.QP(Dmat = Dmat, dvec = dvec, Amat = Amat, bvec = bvec, meq = 1)
    W[i, ] <- sol$solution
  }
  W
}

eval_rss <- function(X_eval, Z) {
  W <- project_simplex_qp(X_eval, Z)
  Xhat <- W %*% Z
  sum((X_eval - Xhat)^2)
}

support_min_ref <- function(X_eval, Z, tau = 0.3) {
  W <- project_simplex_qp(X_eval, Z)
  min(colSums(W > tau))
}

match_perm <- function(Zref, Z) {
  K <- nrow(Zref)
  D <- as.matrix(dist(rbind(Zref, Z)))
  D12 <- D[1:K, (K + 1):(2 * K)]
  as.integer(clue::solve_LSAP(D12))
}

drift_summary <- function(Zref, Z) {
  p <- match_perm(Zref, Z)
  d <- sqrt(rowSums((Zref - Z[p, , drop = FALSE])^2))
  c(median = median(d), p90 = as.numeric(quantile(d, 0.90)))
}

# -----------------------------
# Run selection for a matrix X
# -----------------------------
run_one_space <- function(
    X,
    kmin = 6, kmax = 20,
    C_size = 12000, eval_size = 8000, fit_size = 12000,
    R_runs = 8, nrep = 10,
    knn_k = 30, alpha = 1.5,
    frac_C_fit = 0.7, frac_C_eval = 0.5,
    tau_support = 0.3,
    seed = 1L,
    save_Z_runs = TRUE
) {
  Xs <- scale_embed(X)
  n <- nrow(Xs)
  K_grid <- kmin:kmax
  
  C_idx <- fps_indices(Xs, m = C_size, seed = seed)
  rarity <- knn_rarity(Xs, k = knn_k)
  E_idx <- sample_mix(n, C_idx, rarity, m = eval_size, frac_C = frac_C_eval, alpha = alpha, seed = seed + 101)
  X_eval <- Xs[E_idx, , drop = FALSE]
  
  results <- vector("list", length(K_grid))
  names(results) <- as.character(K_grid)
  
  for (K in K_grid) {
    cat(sprintf("K=%d\n", K))
    
    rss_eval_vec <- numeric(R_runs)
    rss_train_vec <- numeric(R_runs)
    Z_list <- vector("list", R_runs)
    
    for (r in seq_len(R_runs)) {
      F_idx <- sample_mix(n, C_idx, rarity, m = fit_size, frac_C = frac_C_fit, alpha = alpha,
                          seed = seed + 1000 * K + r)
      
      fit <- fit_best_AA(Xs[F_idx, , drop = FALSE], K = K, nrep = nrep, seed = seed + 2000 * K + r)
      
      Z_list[[r]] <- fit$Z
      rss_train_vec[r] <- fit$rss_train
      rss_eval_vec[r] <- eval_rss(X_eval, fit$Z)
      
      rm(fit); gc(FALSE)
    }
    
    ref <- which.min(rss_eval_vec)
    Zref <- Z_list[[ref]]
    drift_stats <- vapply(seq_len(R_runs), function(r) drift_summary(Zref, Z_list[[r]]), numeric(2))
    drift_med <- median(drift_stats["median", ])
    drift_p90 <- as.numeric(quantile(drift_stats["p90", ], 0.90))
    
    sup_min <- support_min_ref(X_eval, Zref, tau = tau_support)
    
    results[[as.character(K)]] <- list(
      rss_eval = rss_eval_vec,
      rss_train = rss_train_vec,
      rss_eval_median = median(rss_eval_vec),
      rss_eval_p10 = as.numeric(quantile(rss_eval_vec, 0.10)),
      rss_eval_p90 = as.numeric(quantile(rss_eval_vec, 0.90)),
      rss_train_min = min(rss_train_vec),
      drift_median = drift_med,
      drift_p90 = drift_p90,
      support_min_ref = sup_min,
      ref_run = ref,
      Z_ref = Zref,
      Z_runs = if (save_Z_runs) Z_list else NULL
    )
    
    rm(Z_list); gc(FALSE)
  }
  
  list(
    params = list(
      kmin = kmin, kmax = kmax,
      C_size = C_size, eval_size = eval_size, fit_size = fit_size,
      R_runs = R_runs, nrep = nrep,
      knn_k = knn_k, alpha = alpha,
      frac_C_fit = frac_C_fit, frac_C_eval = frac_C_eval,
      tau_support = tau_support,
      seed = seed,
      save_Z_runs = save_Z_runs
    ),
    eval_indices = E_idx,
    results = results
  )
}

# -----------------------------
# Entrypoint
# -----------------------------

args <- commandArgs(trailingOnly = TRUE)
gamma = as.numeric(args[[1]])
knn = as.numeric(args[[2]])
channel = args[[3]]
pw = as.logical(args[[4]])

indir = "/gstore/data/marioni_group/Carolina/CP2.0/ArchetypeAnalysis/tmp_archetype_outputs/"

# Auto-detect npc from existing supercell files
message("Auto-detecting npc for channel ", channel, " with gamma=", gamma, ", knn=", knn)
pattern <- if(pw) {
  sprintf("supercells-%s_gamma%s_knn%s_npc*_pw.rds", channel, gamma, knn)
} else {
  sprintf("supercells-%s_gamma%s_knn%s_npc*.rds", channel, gamma, knn)
}
sc_files <- Sys.glob(file.path(indir, channel, pattern))
if(pw) {
  # Ensure we only get _pw files
  sc_files <- sc_files[grepl("_pw\\.rds$", sc_files)]
} else {
  # Ensure we exclude _pw files
  sc_files <- sc_files[!grepl("_pw\\.rds$", sc_files)]
}

if(length(sc_files) == 0) {
  stop("No supercell files found matching pattern: ", file.path(indir, channel, pattern))
}
if(length(sc_files) > 1) {
  warning("Multiple supercell files found, using first: ", basename(sc_files[1]))
}

# Parse npc from filename
sc_file <- sc_files[1]
npc_match <- regexec("_npc([0-9]+)", basename(sc_file))
npc_str <- regmatches(basename(sc_file), npc_match)[[1]][2]
npc <- as.numeric(npc_str)
message("Auto-detected npc = ", npc)

if(pw){
  supercells <- readRDS(paste0(indir,channel,"/supercells-",channel,
                               "_gamma",gamma,"_knn",knn,"_npc",npc,"_pw.rds"))
  pca <- readRDS(paste0(indir,channel,"/cell_pca_centered_pw.rds"))
  scores <- pca$scores
} else{
  supercells <- readRDS(paste0(indir,channel,"/supercells-",channel,
                               "_gamma",gamma,"_knn",knn,"_npc",npc,".rds"))
  scores <- read_parquet(paste0(indir,channel,"/cell_pca_centered.parquet"))
}

# Aggregate to mean
aggregated <- supercell_GE(t(scores[,1:npc]), supercells$membership)

# Write metacell coordinates
if(pw){
  write_parquet(as.data.frame(t(as.matrix(aggregated))),
                paste0(indir,channel,'/supercells-',channel,'_gamma',gamma,
                       '_knn',knn,'_npc',npc,'_pw.parquet'))
} else{
  write_parquet(as.data.frame(t(as.matrix(aggregated))),
                paste0(indir,channel,'/supercells-',channel,'_gamma',gamma,
                       '_knn',knn,'_npc',npc,'.parquet'))
}

message("Supercell PC coordinates saved to parquet file")
rm(pca); gc(FALSE)

res <- run_one_space(t(as.matrix(aggregated)))

if(pw){
  saveRDS(res,paste0(indir,channel,"/archetype_robust_runs_gamma",gamma,
                     "_knn",knn,"_npc",npc,"_pw.rds"))
} else{
  saveRDS(res,paste0(indir,channel,"/archetype_robust_runs_gamma",gamma,
                     "_knn",knn,"_npc",npc,".rds"))
}
