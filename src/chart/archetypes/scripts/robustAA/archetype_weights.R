suppressPackageStartupMessages({
  library(archetypes)
  library(arrow)
})

# =============================
# Command line arguments
# =============================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript archetype_weights.R <channel> [use_pw] [gamma] [knn] [k]\n",
       "  channel: Channel name (e.g., DAPI1, Golgin97, Fibrillarin)\n",
       "  use_pw:  Optional, TRUE or FALSE (default: FALSE)\n",
       "  gamma, knn: which swept config to take the metacells from ",
       "(default 10 10)\n",
       "  k:       Optional, how many archetypes; the k.txt the selectk ",
       "step wrote is read when this is left out")
}

channel <- args[1]
use_pw <- if (length(args) >= 2) as.logical(args[2]) else FALSE
sel_gamma <- if (length(args) >= 3) args[3] else "10"
sel_knn <- if (length(args) >= 4) args[4] else "10"
sel_k <- if (length(args) >= 5) as.integer(args[5]) else NA_integer_
if (length(args) >= 5 && is.na(sel_k)) {
  stop("k given as '", args[5], "', which is not a whole number")
}

suffix <- if (use_pw) "_pw" else ""

#: How many fits to take the best of.  More than the robust runs use,
#: because this one is fitted once and kept.
NREP <- 20

#: Fixed so that two runs of this step on the same metacells give the
#: same archetypes.  stepArchetypes starts from a random simplex.
SEED <- 1L

# =============================
# Where the work happens
# =============================
# Every step of this stage reads and writes inside one directory per
# channel.  'chart archetypes' sets it; by hand, export it once.
sc_dir <- Sys.getenv("CHART_ARCHETYPE_DIR")
if (sc_dir == "") {
  stop("Set CHART_ARCHETYPE_DIR to this channel's directory, or run this ",
       "through 'chart archetypes'")
}

# The weight distributions are looked at to judge whether K was sensible,
# and nothing downstream reads them, so they go to the reports root.
report_dir <- Sys.getenv("CHART_ARCHETYPE_REPORT_DIR")
if (report_dir == "") report_dir <- sc_dir
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

# =============================
# How many archetypes
# =============================
# Either given here, or taken from what the selectk step decided.  Read
# at this point rather than passed in, because 'chart archetypes' builds
# every command before running any of them, and in a whole run k.txt does
# not exist yet when it does that.
if (is.na(sel_k)) {
  k_file <- file.path(sc_dir, sprintf("k%s.txt", suffix))
  if (!file.exists(k_file)) {
    stop("No K given and no ", basename(k_file), " in ", sc_dir, ". Run the ",
         "selectk step first, or set archetypes.k for this channel.")
  }
  sel_k <- as.integer(readLines(k_file, warn = FALSE)[1])
  if (is.na(sel_k)) {
    stop(k_file, " does not hold a whole number, so the number of ",
         "archetypes cannot be read from it")
  }
  cat("Using K =", sel_k, "from", k_file, "\n")
} else {
  cat("Using the K given on the command line:", sel_k, "\n")
}
if (sel_k < 2) {
  stop("K is ", sel_k, ", and archetype analysis needs at least two")
}

# =============================
# The metacells to fit
# =============================
# Taken from where the npc step wrote its decision, rather than read back
# out of a file name.  A run left over from an earlier npc leaves a file
# that fits the naming pattern as well as the current one does, and
# picking between them by name picks the lower number, not the current.
npc_file <- file.path(sc_dir, "npc.txt")
if (!file.exists(npc_file)) {
  stop("No npc.txt in ", sc_dir, ". Run the npc step first.")
}
npc <- suppressWarnings(as.integer(readLines(npc_file, warn = FALSE)[1]))
if (is.na(npc) || npc < 1) {
  stop(npc_file, " does not hold a number of components")
}

coord_path <- file.path(sc_dir, sprintf(
  "supercells-%s_gamma%s_knn%s_npc%d%s.parquet",
  channel, sel_gamma, sel_knn, npc, suffix))
if (!file.exists(coord_path)) {
  stop("npc.txt says ", npc, " components, but there are no metacell ",
       "coordinates for that at ", basename(coord_path), ". Run the ",
       "metacells step for gamma ", sel_gamma, " knn ", sel_knn, ", or check ",
       "which npc its files name.")
}
cat("Metacells:", basename(coord_path), "with npc =", npc, "from npc.txt\n")

# The metacells step writes these components and no more, so the cut does
# nothing to anything it wrote.  It is here for files from before that was
# true, which held every component whatever their name said.
X <- as.matrix(read_parquet(coord_path))
if (ncol(X) < npc) {
  stop(basename(coord_path), " has ", ncol(X), " components, fewer than the ",
       npc, " npc.txt asks for")
}
X <- X[, seq_len(npc), drop = FALSE]

if (nrow(X) <= sel_k) {
  stop("K is ", sel_k, " but there are only ", nrow(X), " metacells to ",
       "place them among")
}

# =============================
# Fit once, on every metacell
# =============================
# The robust runs fitted repeatedly on subsamples to see which K held up.
# This fits the K they settled on to all of the metacells, which is the
# model the weights come from.
#
# Left unscaled, as the analysis this comes from had it: the single cell
# space predicted on below is unscaled too, so the archetypes have to be
# in the same units.  Note that the robust runs scale, so the K was
# chosen in a different space from the one it is used in.
cat("Fitting", sel_k, "archetypes to", nrow(X), "metacells, best of", NREP,
    "\n")
set.seed(SEED)
model <- bestModel(stepArchetypes(X, k = sel_k, nrep = NREP, verbose = FALSE))

if (nrow(parameters(model)) != sel_k || anyNA(parameters(model))) {
  stop("The fit came back with ", nrow(parameters(model)), " archetypes of ",
       "the ", sel_k, " asked for, or with gaps in them, so there is no ",
       "model to weight the cells against")
}

# =============================
# Weights for every cell
# =============================
if (use_pw) {
  pca <- readRDS(file.path(sc_dir, "cell_pca_centered_pw.rds"))$scores
} else {
  pca <- as.data.frame(read_parquet(file.path(sc_dir,
                                              "cell_pca_centered.parquet")))
}

# The pca step puts what identifies a cell in the last four columns.
metadata <- pca[, tail(colnames(pca), 4), drop = FALSE]
weights <- predict(model, pca[, seq_len(npc), drop = FALSE])
colnames(weights) <- paste0("Archetype", seq_len(sel_k))
cat("Weighted", nrow(weights), "cells\n")

out_name <- sprintf("robust_archweights-%s_k%s_gamma%s_knn%s_npc%s%s",
                    channel, sel_k, sel_gamma, sel_knn, npc, suffix)

pdf(file.path(report_dir, paste0(out_name, ".pdf")))
for (a in colnames(weights)) {
  plot(density(weights[, a]), main = paste(channel, a))
}
invisible(dev.off())
cat("Saved weight distributions to:",
    file.path(report_dir, paste0(out_name, ".pdf")), "\n")

out_path <- file.path(sc_dir, paste0(out_name, ".parquet"))
write_parquet(cbind.data.frame(as.data.frame(weights), metadata), out_path)
cat("Wrote", out_path, "\n")
