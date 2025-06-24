# feature_group_ingestion.R
# 163-165

library(tidyverse)

clinData <- read.csv('./data/raw/clinicalData_clean.csv')
imagData <- read.csv('./data/raw/imagingFeatures.csv')
featureGroups <- read.csv('./interim/imFeatures_and_feature_citations.csv')

data <- clinData %>% inner_join(imagData, by = "Patient.ID")

