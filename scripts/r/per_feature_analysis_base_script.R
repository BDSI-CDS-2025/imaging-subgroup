# per_feature_analysis.R
# Allows you to perform analysis for each subgroup

# for relative paths
library(here)

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

# iterate over each feature group
for (group in groups) {
    file_path <- here("data", "raw", group)
    data <- read.csv(file_path)

    # perform data analysis on each feature group here
}