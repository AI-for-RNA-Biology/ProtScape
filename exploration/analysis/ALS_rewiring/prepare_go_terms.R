# Export direct human GO annotations for the module pathway embeddings.
# Run from the repository root with org.Hs.eg.db, GO.db and yaml installed.
suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(GO.db)
})

paths <- yaml::read_yaml("configs/paths.yaml")
output <- path.expand(paths$als_rewiring_go_terms_csv)
genes_by_term <- as.list(org.Hs.egGO2EG)
descriptions <- Term(GOTERM)

rows <- lapply(names(genes_by_term), function(go_id) {
  symbols <- suppressMessages(mapIds(
    org.Hs.eg.db, keys = unlist(genes_by_term[go_id]),
    column = "SYMBOL", keytype = "ENTREZID"
  ))
  symbols <- symbols[!is.na(symbols) & nzchar(symbols)]
  if (!length(symbols)) return(NULL)
  data.frame(GO_id = go_id, Description = descriptions[go_id],
             Gene_Symbols = paste(symbols, collapse = ", "),
             stringsAsFactors = FALSE)
})

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), output, row.names = FALSE)
message("Saved ", output, " (", length(rows), " GO terms)")
message("org.Hs.eg.db ", packageVersion("org.Hs.eg.db"),
        "; GO.db ", packageVersion("GO.db"))
