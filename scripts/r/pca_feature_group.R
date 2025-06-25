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

# create dataframe object that will store loading information for top three
loading_colnames <- c("group", "variable", "loading_factor")
loading <- data.frame(matrix(nrow = 0, ncol = length(loading_colnames)))
colnames(loading) = loading_colnames
print(loading)

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
    xlab("Explained Variance (%)") +
    ylab("Number of Principal Components") +
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
    Sample = rownames(data), # in case we would like to label each point with its row
    ER = joined_data$ER, # for coloring
    X = pca$x[, 1],
    Y = pca$x[, 2])
  scatter <- ggplot(data = pca_data,
    aes(x = X, y = Y, label = Sample, color=ER)) + 
    #geom_text() + # add back if you would like to label point with row
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

  # plot the loading factors just for one component
  loading_scores <- pca$rotation[, 1]
  var_scores <- abs(loading_scores) # only care about magnitude
  var_score_ranked <- sort(var_scores, decreasing = TRUE)
  top_3_variables <- names(var_score_ranked[1:3])
  print(group_name)
  print(top_3_variables)

  for (i in 1:3){
    temp <- data.frame(group = group_name,
      variable = top_3_variables[i],
      loading_factor = var_score_ranked[i])
    loading <- rbind(loading, temp)
    #loading[nrow(loading) + 1] = c(group_name, top_3_variables[i], var_score_ranked[i])
  }

  # df[nrow(df) + 1,] = c("v1","v2")
  
}

# write the loading factor results to a .csv file
file_path_joined <- here("results", "reports", "top_three_loading_factors_by_sugroup.csv")
#print(loading)
write.csv(loading, file_path_joined)