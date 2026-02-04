# Brain MRI Classification: Model Comparison

This project implements and compares two deep learning approaches for classifying brain MRI scans into three categories: no tumor, glioblastoma, and metastasis.

## Models

### Model 1: End-to-End CNN
- Convolutional neural network with MLP decoder
- Trained end-to-end with cross-entropy loss
- Direct supervised learning approach

### Model 2: VAE + MLP Classifier
- Two-stage approach:
  1. Variational autoencoder (VAE) trained with reconstruction + KL divergence loss
  2. MLP classifier trained on frozen VAE embeddings with cross-entropy loss
- Representation learning followed by supervised classification

## Project Structure

```
Research_work/
├── config.py                 # Configuration and hyperparameters
├── data_loader.py           # Data loading and preprocessing
├── model1_cnn.py            # Model 1 architecture
├── model2_vae.py            # Model 2 architecture (VAE + MLP)
├── train_model1.py          # Training script for Model 1
├── train_model2.py          # Training script for Model 2
├── evaluate.py              # Evaluation script
├── visualize_results.py     # Visualization script
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Data directory (create this)
│   ├── no_tumor/           # PNG images of no tumor
│   ├── glioblastoma/       # PNG images of glioblastoma
│   └── metastasis/         # PNG images of metastasis
├── saved_models/           # Trained model checkpoints (auto-created)
└── results/                # Results and plots (auto-created)
    └── plots/
```

## Data Preparation

Organize your brain MRI PNG images in the following structure:

```
data/
├── no_tumor/
│   └── *.png
├── glioblastoma/
│   └── *.png
└── metastasis/
    └── *.png
```

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Full Pipeline)

Run the complete pipeline (train both models, evaluate, and visualize):

```bash
python main.py
```

This will:
1. Load and split the data (70% train, 15% val, 15% test)
2. Train Model 1 (End-to-end CNN)
3. Train Model 2 (VAE, then MLP classifier)
4. Evaluate both models on the test set
5. Generate comparison visualizations

### Individual Scripts

You can also run individual components:

```bash
# Train only Model 1
python train_model1.py

# Train only Model 2
python train_model2.py

# Evaluate both models (requires trained models)
python evaluate.py

# Generate visualizations (requires evaluation results)
python visualize_results.py
```

## Configuration

Edit `config.py` to customize:
- Image size and normalization
- Train/val/test split ratios
- Batch size and number of workers
- Number of training epochs
- Learning rates and weight decay
- Model architecture parameters
- Random seed for reproducibility

## Evaluation Metrics

The project computes and compares:
- **Accuracy**: Overall classification accuracy
- **AUROC**: Area under ROC curve (one-vs-rest, macro-averaged)
- **F1-Score**: F1 score (macro-averaged and per-class)
- **Confusion Matrix**: Detailed classification breakdown

## Visualizations

The following plots are automatically generated in `results/plots/`:

1. **training_curves.png**: Training and validation loss/accuracy for both models
2. **vae_training_curves.png**: VAE reconstruction and KL divergence losses
3. **metrics_comparison.png**: Bar chart comparing accuracy, AUROC, and F1-score
4. **f1_per_class.png**: Per-class F1-score comparison
5. **confusion_matrices.png**: Side-by-side confusion matrices
6. **roc_curves.png**: ROC curves for each class (one-vs-rest)
7. **summary_table.png**: Complete comparison table

## Output Files

After running the pipeline:

### Models
- `saved_models/model1_cnn.pth`: Trained Model 1
- `saved_models/model2_vae.pth`: Trained VAE encoder
- `saved_models/model2_classifier.pth`: Trained MLP classifier

### Results
- `results/model1_results.json`: Detailed Model 1 evaluation results
- `results/model2_results.json`: Detailed Model 2 evaluation results
- `results/model1_results_history.json`: Model 1 training history
- `results/model2_results_history.json`: Model 2 training history

### Plots
- All visualization plots in `results/plots/`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- scikit-learn 1.3+
- matplotlib 3.7+
- seaborn 0.12+
- Pillow 10.0+
- tqdm 4.66+
- NumPy 1.24+

## Notes

- The code automatically detects and uses GPU if available
- Training progress is shown with progress bars
- Best models are saved based on validation performance
- Random seed is set for reproducibility
- Data augmentation is applied during training (rotation, flipping, translation)

## License

This project is for research and educational purposes.
