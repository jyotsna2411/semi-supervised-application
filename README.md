# Predicting Amphibian Survival Based on Environmental Features: Using Semi-Supervised Learning

## Overview

This project aims to predict amphibian presence in water reservoirs leveraging GIS, satellite, and environmental impact assessment reports. The dataset, originating from two road projects in Poland, integrates information about water reservoirs with data on amphibian species. The primary objective is to comprehend amphibian habitat preferences and contribute insights for conservation efforts in areas impacted by road development.

## Dataset

- **Instances:** 189
- **Attributes:** 22 (18 quantitative, 4 qualitative)
- **Source:** GIS, satellite, and environmental impact assessment reports

### Attribute Categories:

1. **Geographical Information:**
   - Location, Altitude, Type of water reservoir.

2. **Environmental Factors:**
   - Temperature, Precipitation, Humidity.

3. **Amphibian Species Presence:**
   - Labels for 7 different amphibian species.

## Preprocessing

Data preprocessing is a vital step in transforming raw data into a structured format suitable for analysis and machine learning. The dataset, comprising 189 instances and 22 attributes, underwent the following steps:

1. **Combination of Datasets:**
   - Merged two datasets: one containing water reservoir information and the other containing amphibian species information.

2. **Handling Missing Values:**
   - The dataset has no missing values.

3. **Data Cleaning:**
   - Ensured consistency and eliminated noise in the dataset.

## Research Paper

For deeper insights into the study, please refer to the research paper titled "Predicting presence of amphibian species using features obtained from GIS and satellite images" by Marcin Blachnik, Marek Sołtysiak, and Dominika Dąbrowska.

## Usage

1. **Clone the repository:**
   ```bash
   https://github.com/jyotsna2411/semi-supervised-application.git
