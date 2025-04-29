# Protein Target Analysis and Synthetic Data Generation

## Project Overview
This project processes the **ChEMBL** database to analyze protein targets and set up the groundwork for generating synthetic data. It also incorporates **mutation analysis** using UniProt data to investigate wild-type and mutation-specific protein interactions.

### Key Features
- **Data Cleaning**: Flexible filtering of ChEMBL data, including options to focus on specific assay types, organisms, and mutation details.
- **Descriptors Calculation**: RDKit and Mordred are used for molecular descriptor generation.
- **Mutation Analysis**: Processes mutation data using UniProt to add insights such as known mutations, accession codes, and population data.
- **Machine Learning Models**: Supports model training with configurable algorithms like SVM, Random Forest, and logistic regression.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FedericaSantarcangelo/Protein.git
   cd your-project

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Output
- **wild type**: for wild-type proteins.
- **mutation_target**: data focusing on mutation-specific proteins.
- **mixed**: all the other data

The cleaned data is organized into three categories, each with subfolders for different quality levels, reflecting different degrees of confidence.
### Quality
- **Quality 1**: highest confidence level,with well-characterized data.
- **Quality 2**: medium confidence, possibly with minor uncertainties.
- **Quality 3**: lowest confidence, requires additional verification.

## Model Training
The pipeline supports training of machine learning models using the processed data.
### Input Format Notice

> **Note**: The input file used for model training is **not** the same as the one produced during the data cleaning phase.

Below is the expected structure (columns may vary slightly depending on configuration):

| Column Name               | Description                                                 |
|---------------------------|-------------------------------------------------------------|
| Molecule ChEMBL ID        | Unique identifier for each molecule                        |
| Smiles (RDKit Mol)        | Molecular structure in RDKit Mol format                    |
| MACCS_sim_score           | Similarity score using MACCS keys (optional)               |
| ECFP4_sim_score           | Similarity score using ECFP4 fingerprints (optional)       |
| MCSS_rdkit_sim_score      | Similarity score from RDKit's MCSS method (optional)       |
| Standard Type             | Measurement type (e.g., IC50, Ki)                          |
| Standard Relation         | Comparison symbol (e.g., '=', '>', '<')                   |
| Standard Value            | Activity value (numeric)                                   |
| Standard Units            | Measurement unit (e.g., nM)                                |
| Document ChEMBL ID        | Reference document ID from ChEMBL                         |
| Smiles                    | Canonical SMILES string                                    |


### Model Selection and Training Pipeline
Once the dataset is prepared, the training pipeline follows these steps:
1. Dimensionality Reduction (PCA): Principal Component Analysis is applied to reduce feature dimensionality and improve model efficiency.
2. Hyperparameter Tuning: for each model, a grid search is performed using GridSearchCV to find the best configuration. The scoring metric depends on the task (e.g., RMSE, MAE, or R²).
3. Model Retraining: after the best configuration is found, the model is retrained on the full training set using the optimal hyperparameters.
4. Final Testing: the retrained model is evaluated on a held-out test set to assess generalization.

### Supported Models
The following regression models are supported in the training pipeline:
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Multi-layer Perceptron (MLP)
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor (KNN)
- XGBoost Regressor

Each model is trained and optimized independently using its own hyperparameter grid.