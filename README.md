# Project Title

Official implementation of "A Physics-Guided Framework for False Data Injection Attack Generation, Validation, and Knowledge-Distillation-Based Detection in Smart Grids".

## Overview

This repository contains the research code accompanying "A Physics-Guided Framework for False Data Injection Attack Generation, Validation, and Knowledge-Distillation-Based Detection in Smart Grids" for generating and evaluating synthetic false data injection (FDI) attacks in power systems.

The repository consists of separate Python scripts for OPF-based normal operating data generation, machine learning-based FDI attack generation, AC power flow-based physical consistency validation, statistical analysis, data visualization, and teacher–student-based attack detection. Experiments use the IEEE 14-bus and IEEE 118-bus benchmark systems.

Each script is run individually, with its own input files and parameter settings.

This repository provides code for:
- Generating normal operating data for IEEE benchmark systems.
- Generating false data injection (FDI) attacks.
- Evaluating the physical consistency and statistical properties of
  the generated data.
- Training and evaluating detection models.

## Repository Structure

The following scripts are used for the experiments reported in the paper:

| Script | Description |
| --- | --- |
| `NormalDataGen.py` | Generates normal operating data using optimal power flow (OPF). |
| `FDI_Generator_IEEE14.py` | Generates synthetic FDI attack data for the IEEE 14-bus system. |
| `FDI_Generator_IEEE118.py` | Generates synthetic FDI attack data for the IEEE 118-bus system. |
| `flow_equation.py` | Evaluates the physical consistency of the data using AC power flow equations. |
| `data_statistic.py` | Performs statistical analysis of the generated data. |
| `data_visualize_forsubmission.py` | Produces visualizations for the paper. |
| `teacher_student_model.py` | Trains and evaluates teacher–student models for FDI attack detection. |

### Baseline Implementation

`SAGAN_model.py` implements the SA-GAN baseline based on "A new ac false data injection attack method without network information", for comparison with the proposed method.

### Other Scripts

The remaining scripts contain exploratory code and earlier experimental versions developed during the study. They are not required to reproduce the results reported in the paper.
## Requirements

- Python 3.14
- scikit-learn==1.9.1
  pandapower==3.5.6
  xgboost==3.4.1
  lightgbm==4.7.0
  matplotlib==3.11.2
  seaborn==0.13.2
  torch==2.14.1

Install the required packages:

pip install -r requirements.txt

## Usage

For each of the code, adjust the input parameters (e.g., input file name and output file name, number of data point to be generated) as desired.

### 1. Generate Normal Operating Data

python NormalDataGen.py

### 2. Generate FDI Attacks

Run the script for the desired benchmark:

python FDI_Generator_IEEE14.py
python FDI_Generator_IEEE118.py

### 3. Validate the Generated Data

Physical Consistency:
python flow_equation.py

Statistical Comparison:
python data_statistic.py

Data Visualization:
python data_visualize_forsubmission.py

### 4. Train and Evaluate Detection Models

python teacher_student_model.py

## Citation

The accompanying manuscript is currently in preparation.
Citation information will be added once a preprint or publication is available.

## License

This project is licensed under the MIT License. See the [LICENSE] file for details.

## Contact

For questions, please contact Qiaodan Luo at qiaodan.luo@dal.ca or qiaodan.luo@mail.mcgill.ca, or open a GitHub issue.