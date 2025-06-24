# pca_feature_group.R
# performs PCA analysis on each feature group

library(here)
library(tidyverse)

# the name of the file for each subgroup
groups <- c(#"Combining_Tumor_and_FGT_Enhancement.csv",
            "Tumor_Size_and_Morphology.csv",
            "Tumor_Enhancement_Texture.csv",
            "Tumor_Enhancement_Spatial_Heterogeneity.csv",
            "Tumor_Enhancement_Variation.csv",
            "Breast_and_FGT_Volume_Features.csv",
            "FGT_Enhancement.csv",
            "Tumor_Enhancement.csv",
            "FGT_Enhancement_Texture.csv",
            "FGT_Enhancement_Variation.csv")

for (group in groups) {
    file_path <- here("data", "raw", group)
    data <- read.csv(file_path)
    groupName <- substr(group, 1, nchar(group) - 4)

    # perform data analysis on each feature group here
    pcdData <- data %>%
      select(where(is.numeric)) %>% # only use columns that are of numeric data type
      sapply(function(x) ifelse(is.na(x), mean(x, na.rm=TRUE), x)) # replace with mean if missing value

    pca <- prcomp(pcaData, scale=TRUE)

    # PCA scatter plot
    p <- as.data.frame(pca$x) %>%
        ggplot(aes(x = PC1, y = PC2, color = PC3)) +
        geom_point() +
        ggtitle(group)
    ggsave(
        filename = here("results", "figures", "pca", paste0("pca_plot_", gsub(" ", "_", groupName), ".png")),
        plot = p
    )
    

    # calculate explained variance percentages
    percent <- round(100 * pca$sdev^2 / sum(pca$sdev^2), 1)
    percent_df <- data.frame(PC = paste0("PC", seq_along(percent)), Percent = percent)
    
    # histogram of explained variance percentages
    p2 <- ggplot(percent_df, aes(x = Percent)) +
        geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
        xlab("Explained Variance (%)") +
        ylab("Number of Principal Components") +
        ggtitle(paste("PCA Variance Histogram -", groupName))

    ggsave(
        filename = here("results", "figures", "pca", paste0("pca_hist_", gsub(" ", "_", groupName), ".png")),
        plot = p2
    )
    }