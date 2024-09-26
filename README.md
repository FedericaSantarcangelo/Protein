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

