# PCA-GAT
PCA-GAT
PCA-GAT: Process Constraint-Aware Graph Attention Network for Manufacturing Process Planning Recommendation
1. Overview
PCA-GAT (Process Constraint-Aware Graph Attention Transformer) is a knowledge graph enhanced recommendation model tailored for manufacturing process planning. It integrates process constraints into graph attention learning to recommend valid and optimal machining process plans while preserving high recommendation accuracy.
2. Project Structure
plaintext
PCA-GAT/
├── src/
│   ├── checkpoints/          # Model checkpoints (generated during training)
│   ├── data/                 # Machining dataset and preprocessing scripts
│   ├── logs/                 # Training logs
│   ├── config_pcagat.py      # Global configuration (hyperparameters, paths)
│   ├── dataset_machining.py  # Dataset loader and preprocessing for machining KG
│   ├── explainer.py          # Explainability analysis module
│   ├── model_pcagat.py       # Core PCA-GAT model implementation
│   ├── run_ablation.py       # Ablation study (RQ2)
│   ├── run_explainability.py # Case study & explainability analysis (RQ6)
│   ├── run_pcagat.py         # Main training script for PCA-GAT
│   ├── run_sensitivity.py    # Parameter sensitivity analysis (RQ4)
│   ├── run_sparsity_experiment.py # Sparsity robustness experiment
│   └── utils_pcagat.py       # Evaluation metrics and helper functions
├── .gitignore
├── LICENSE
└── README.md
3. Experimental Correspondence
表格
Paper Section	Corresponding Script	Research Question
4.1 Experimental Setup	config_pcagat.py, dataset_machining.py	Experimental setup and data preparation
4.2 Overall Performance (RQ1)	run_sparsity_experiment.py	Overall performance comparison with baselines
4.3 Ablation Study (RQ2)	run_ablation.py	Ablation study on module effectiveness
4.4 Constraint Analysis (RQ3)	utils_pcagat.py (constraint metrics)	Constraint contribution and validity analysis
4.5 Parameter Sensitivity (RQ4)	run_sensitivity.py	Hyperparameter sensitivity analysis
4.6 Case Study & Explainability (RQ6)	run_explainability.py, explainer.py	Case study and explainability analysis
4. Quick Start
4.1 Environment Setup
bash
运行
# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
4.2 Dataset Preparation
Place the machining dataset into src/data/machining_process/ (including train.txt, test.txt, kg_final.txt, constraint_rules.json, etc.).
4.3 Run PCA-GAT
bash
运行
cd src
python run_pcagat.py
4.4 Run Other Experiments
bash
运行
# Ablation study (RQ2)
python run_ablation.py

# Parameter sensitivity analysis (RQ4)
python run_sensitivity.py

# Explainability & case study (RQ6)
python run_explainability.py

# Sparsity experiment & baseline comparison
python run_sparsity_experiment.py
5. Key Features
Constraint-Aware GAT: Integrates manufacturing process constraints into graph attention to avoid invalid recommendations.
Knowledge Graph Enhancement: Leverages KG to enrich item representations with semantic information.
Comprehensive Experimental Pipeline: Supports ablation studies, sparsity analysis, parameter sensitivity, and explainability analysis.
Reproducibility: All experiments are reproducible with fixed random seeds and standardized training pipelines.
6. Citation
If you use this code in your research, please cite our paper:
bibtex
@article{YourPaper2025,
  title={PCA-GAT: Process Constraint-Aware Graph Attention Network for Manufacturing Process Planning Recommendation},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
7. License
This project is licensed under the Apache-2.0 License.
