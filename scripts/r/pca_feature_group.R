# pca_feature_group.R
# performs PCA analysis on each feature group

library(here)
library(tidyverse)

# the name of the file for each subgroup
groups <- c("Combining_Tumor_and_FGT_Enhancement.csv",
            "Tumor_Size_and_Morphology.csv",
            "Tumor_Enhancement_Texture.csv",
            "Tumor_Enhancement_Spatial_Heterogeneity.csv",
            "Tumor_Enhancement_Variation.csv",
            "Breast_and_FGT_Volume_Features.csv",
            "FGT_Enhancement.csv",
            "Tumor_Enhancement.csv",
            "FGT_Enhancement_Texture.csv",
            "FGT_Enhancement_Variation.csv")

file_path_joined <- here("data", "raw", "joined_data.csv")
joined_data <- read.csv(file_path_joined)
joined_data <- joined_data %>% mutate(ER = as.factor(ER))

# column names for dataframes storing information about loading factors
loading_colnames <- c("group", "variable", "loading_factor")

# create dataframe object that will store loading information for top three
loading_top_three <- data.frame(matrix(nrow = 0,
                                       ncol = length(loading_colnames)))
colnames(loading_top_three) <- loading_colnames

# create a dataframe object that will store loading information for
# principal components that explain 90% of variance
loading_ninety <- data.frame(matrix(nrow = 0,
                                    ncol = length(loading_colnames)))
colnames(loading_ninety) <- loading_colnames

# a dataframe that is going to be PC1 for each group
patient_component <- data.frame(matrix(nrow = 922,
                                       ncol = 0))
patient_component$Patient.ID <- joined_data$Patient.ID

# the number of principal component coordinates to output
N = 3

for (group in groups) {
  file_path <- here("data", "raw", group)
  data <- read.csv(file_path)
  group_name <- substr(group, 1, nchar(group) - 4)

  # perform data analysis on each feature group here
  pcd_data <- data %>%
    select(where(is.numeric)) %>%
    # only use columns that are of numeric data type
    sapply(
      function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x)
    ) # replace with mean if missing value

  pca <- prcomp(pcd_data, scale = TRUE)
  # pca$x: contains the PCs for the graph
  # pca$sdev: how much variation in the data each PC accounts for
  # pca$rotation: the loading factors

  # scree plot: calculate explained variance percentages
  percent <- round(100 * pca$sdev ^ 2 / sum(pca$sdev ^ 2), 1)
  percent_df <- data.frame(
    PC = paste0("PC", seq_along(percent)),
    Percent = percent
  )

  # histogram of explained variance percentages
  scree <- ggplot(percent_df, aes(x = PC, y = Percent)) +
    geom_col() +
    xlab("Principal Component") +
    ylab("Explained Variance (%)") +
    ggtitle(paste("PCA Variance Histogram -", group_name))

  ggsave(
    filename = here(
      "results", "figures", "pca",
      paste0("pca_scree_", gsub(" ", "_", group_name), ".png")
    ),
    plot = scree
  )

  # PCA scatter plot
  pca_data <- data.frame(
                         # in case we would like to label
                         # each point with its row
                         Sample = rownames(data),
                         ER = joined_data$ER, # for coloring
                         X = pca$x[, 1],
                         Y = pca$x[, 2])
  scatter <- ggplot(data = pca_data,
                    aes(x = X, y = Y, label = Sample, color = ER)) +
                    #geom_text() + # add back to label
                    geom_point() +
                    xlab(paste("PC1 - ", percent[1], "%", sep = " ")) +
                    ylab(paste("PC2 - ", percent[2], "%", sep = " ")) +
                    theme_bw() + # makes the graph's background white
                    ggtitle(paste("PCA Plot - ", group_name))

  ggsave(
    filename = here(
      "results", "figures", "pca",
      paste0("pca_scatter_", gsub(" ", "_", group_name), ".png")
    ),
    plot = scatter
  )

  # the loading factors just the first component
  loading_scores <- pca$rotation[, 1]
  var_scores <- abs(loading_scores) # only care about magnitude
  var_score_ranked <- sort(var_scores, decreasing = TRUE)
  variable_names <- names(var_score_ranked)

  for (i in 1:3) {
    temp <- data.frame(group = group_name,
                       variable = variable_names[i],
                       loading_factor = var_score_ranked[i])
    loading_top_three <- rbind(loading_top_three, temp)
  }

  # add to data frame to explain up to 90% of the variance
  total <- 0
  i <- 1

  while (total < 0.9) {
    temp <- data.frame(group = group_name,
                       variable = variable_names[i],
                       loading_factor = var_score_ranked[i])
    i <- i + 1
    total <- total + var_score_ranked[i]
    loading_ninety <- rbind(loading_ninety, temp)
  }

  # add all features to the frame
  for (i in 1:N) {
    col_name <- paste0("PC", i, "_", group_name)
    # use double brackets to access the value in col_name as the column name
    patient_component[[col_name]] <- pca$x[, i]
  }

  
}

# write the loading factor results to a .csv file
file_path_joined_top_three <- here("results",
                                   "reports",
                                   "top_three_loading_factors_by_subroup.csv")
write.csv(loading_top_three, file_path_joined_top_three, row.names = FALSE)
file_path_joined_ninety <- here("results",
                                "reports",
                                "ninety_percent_factors_by_subgroup.csv")
write.csv(loading_ninety, file_path_joined_ninety, row.names = FALSE)

# write the PC data to a csv
file_path_joined_pc <- here("data",
                            "interim",
                            "pc1_to_3_by_feature_group_for_patients.csv")
write.csv(patient_component, file_path_joined_pc, row.names = FALSE)