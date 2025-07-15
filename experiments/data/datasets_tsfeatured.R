if (!requireNamespace("tsfeatures", quietly = TRUE)) {
    install.packages("tsfeatures", repos = "https://cran.r-project.org", dependencies = TRUE)
}
library(tsfeatures)

# ============================================================================
# 1) Rutas de entrada y salida ---------------------------------------------
# ============================================================================
raw_root <- "experiments/data/raw"        # donde están los CSV originales
out_root <- "experiments/data/tsfeatured" # destino de los CSV con features
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Lista de datasets = nombres de carpetas dentro de raw_root
datasets <- list.dirs(raw_root, recursive = FALSE, full.names = FALSE)

# ============================================================================
# 2) Bucle principal --------------------------------------------------------
# ============================================================================
for (dset in datasets) {
  message("Procesando ", dset, " …")

  # Asegura carpeta de salida
  out_dir <- file.path(out_root, dset)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # Procesa train y test
  for (split in c("train", "test")) {

    in_csv  <- file.path(raw_root, dset, paste0(split, ".csv"))

    df <- read.csv(in_csv)

    # -----------------------------------------------------------------------
    # 2.1) Separa serie (X) y etiquetas (y) manteniendo el orden
    # -----------------------------------------------------------------------
    X <- df[ , !(names(df) %in% "label")]   # todas las columnas salvo label
    y <- df$label                           # vector/columna de etiquetas

    # -----------------------------------------------------------------------
    # 2.2) Calcula características con tsfeatures
    #  - tsfeatures espera cada serie en una *columna*   (length × n_series)
    #  - nuestras series están en filas                 (n_series × length)
    #  → transponemos con t(...)
    # -----------------------------------------------------------------------
    feats <- tsfeatures( t(as.matrix(X)) )  # mantiene el orden original
    # `feats` es un data-frame con mismas filas que df

    # -----------------------------------------------------------------------
    # 2.3) Combina etiqueta + features y guarda
    # -----------------------------------------------------------------------
    result <- cbind(label = y, feats)       # la etiqueta como primera columna
    out_csv <- file.path(out_dir, paste0(split, ".csv"))
    write.csv(result, out_csv, row.names = FALSE)

    message("  ✔ ", split, " -> ", out_csv)
  }
}
