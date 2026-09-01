library(readxl)

file <- "all_simulations.xlsx"
outdir <- "csv_output"
dir.create(outdir, showWarnings = FALSE)

sheet_names <- excel_sheets(file)

all_data <- lapply(sheet_names, function(s) read_excel(file, sheet = s))
names(all_data) <- sheet_names

# all 30 clean parameters
params <- c("tbend", "shear", "stretch", "stagger", "buckle", "propel", "opening",
            "xdisp", "ydisp", "inclin", "tip", "ax-bend", "shift", "slide", "rise",
            "tilt", "roll", "twist", "h-ris", "h-twi", "phaseW", "ampW", "gammaW",
            "gammaC", "phaseC", "ampC", "minw", "mind", "majw", "majd")

for (p in params) {
  out <- do.call(rbind, lapply(sheet_names, function(s) {
    df <- all_data[[s]]
    df[df[[1]] == p, ]
  }))
  
  out <- out[, -c(2, ncol(out))]        # drop first and last base (edge bp-steps)
  out[[1]] <- sheet_names               # replace label column with sequence IDs
  names(out)[1] <- "sequence"           # rename only the label column, keep step names (e.g. "2-3") as-is
  
  write.table(out, file.path(outdir, paste0(gsub("-", "", p), ".csv")), quote = FALSE, row.names = FALSE, sep = ",")
}
