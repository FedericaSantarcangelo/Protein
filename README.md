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
**Note**: the input file used for model training is **not** the same as the one produced during the data cleaning phase.
It has the following structure (columns nay vary slightly depending on configuration):
- **Molecule ChEMBL ID**: unique molecule identifier
- **Smiles (RDKit Mol)**: molecular structure in RDKit Mol format
- **MACCS_sim_score**: similarity score using MACCS keys (optional)
- **ECFP4_sim_score**: similarity score using ECFP4 fingerprints (optional)
- **MCSS_rdkit_sim_score**: similarity score from RDKit's MCSS method (optional)
- **Standard Type**: type of measurement (e.g. IC50, Ki)
- **Standard Relation**: comparator (e.g. '=', '>', '<')
- **Standard Value**: activity value (numeric)
- **Standard Units**: unit of measurement (e.g. nM)
- **Document ChEMBL ID**: reference document ID
- **Smiles**: canonical SMILES (string representation of the molecule)

**Note**: If any of the similarity columns (*_sim_score) are missing, the training script will automatically compute them using the Smiles column.

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