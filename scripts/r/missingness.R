# missingness.R
# produces a summary report of the missingness statistics.

library(here)
library(dplyr)

file_source <- here("data", "raw", "joined_data.csv")
joined_data <- read.csv(file_source)

# calculate missingness for each column
missingness_report <- data.frame(
  variable = names(joined_data),
  n_missing = sapply(joined_data, function(x) sum(is.na(x))),
  pct_missing = round(100 * sapply(joined_data, function(x) mean(is.na(x))), 2)
)

# sort by percent missing (descending)
missingness_report <- missingness_report %>% arrange(desc(pct_missing))

# write to CSV
missingness_write <- here("results", "reports", "missingness_report.csv")
write.csv(missingness_report, missingness_write, row.names = FALSE)