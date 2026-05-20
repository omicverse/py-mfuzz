#!/usr/bin/env Rscript
# Drive Bioconductor Mfuzz reference on its bundled yeast dataset.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# Outputs (in out_dir):
#   yeast_raw.tsv      raw exprs(yeast) matrix (genes x timepoints, with NA)
#   yeast_filled.tsv   after filter.NA + fill.NA(mode="knn")
#   standardised.tsv   standardise() of the filled matrix
#   mestimate.tsv      estimated fuzzifier m (single value)
#   centers.tsv        mfuzz() cluster centres (c x timepoints)
#   membership.tsv     mfuzz() membership matrix (genes x c)
#   cluster.tsv        mfuzz() hard cluster assignment (genes)
#   dmin.tsv           Dmin() minimum-centroid-distance curve
#   info.tsv           dataset / parameter metadata

suppressPackageStartupMessages({
  library(Mfuzz)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(123)

# --- load the bundled yeast cell-cycle time-course ------------------
data(yeast)

raw <- exprs(yeast)
write.table(raw, file.path(out_dir, "yeast_raw.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# --- NA handling: filter then knn-fill ------------------------------
yeast.r <- filter.NA(yeast, thres = 0.25)
yeast.f <- fill.NA(yeast.r, mode = "knn")
write.table(exprs(yeast.f), file.path(out_dir, "yeast_filled.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# --- standardise ----------------------------------------------------
yeast.s <- standardise(yeast.f)
write.table(exprs(yeast.s), file.path(out_dir, "standardised.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# --- mestimate ------------------------------------------------------
m <- mestimate(yeast.s)
write.table(data.frame(m = m), file.path(out_dir, "mestimate.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- mfuzz soft clustering, c = 16 ----------------------------------
# With a near-1 fuzzifier m, cmeans needs more than the default 100
# iterations to converge and is sensitive to initialisation, so we take
# the best (lowest within-error) of several seeded runs at iter.max=500
# -- a converged, reproducible reference for the Python comparison.
set.seed(123)
cl <- NULL
for (s in 0:19) {
  set.seed(s)
  cand <- mfuzz(yeast.s, c = 16, m = m, iter.max = 500)
  if (is.null(cl) || cand$withinerror < cl$withinerror) cl <- cand
}
write.table(cl$centers, file.path(out_dir, "centers.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(cl$membership, file.path(out_dir, "membership.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(data.frame(NAME = names(cl$cluster),
                       CLUSTER = as.integer(cl$cluster)),
            file.path(out_dir, "cluster.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- Dmin elbow curve -----------------------------------------------
set.seed(123)
dmin <- Dmin(yeast.s, m = m, crange = seq(4, 24, 4),
             repeats = 3, visu = FALSE)
write.table(data.frame(c = seq(4, 24, 4), dmin = dmin),
            file.path(out_dir, "dmin.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- metadata -------------------------------------------------------
write.table(
  data.frame(
    key = c("n_genes_raw", "n_genes_filled", "n_time", "m", "c"),
    value = c(nrow(raw), nrow(exprs(yeast.s)), ncol(raw), m, 16)
  ),
  file.path(out_dir, "info.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

cat("R reference driver done.\n")
