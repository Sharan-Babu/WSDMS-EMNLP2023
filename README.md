# WSDMS-EMNLP2023: HotCake Fake News Detection Model

A robust BERT-based fake news detection system that leverages advanced natural language processing techniques for accurate classification of news articles.

## Project Structure

```
├── config/                     # Configuration files for different datasets
│   ├── PolitiFact.ini         # Config for PolitiFact dataset
│   ├── GossipCop.ini          # Config for GossipCop dataset
│   └── BuzzNews.ini           # Config for BuzzNews dataset
├── utils/                      # Utility functions
│   ├── utils_misc.py          # Miscellaneous helper functions
│   └── utils_preprocess.py    # Data preprocessing utilities
├── sample_data/               # Sample dataset and processing scripts
│   ├── raw/                   # Raw dataset files
│   ├── process_data.py        # Data processing script
│   └── get_train_test_split.py# Train-test split utility
├── HotCakeModels.py          # Core model architecture
├── bert_model.py             # BERT model implementations
├── data_loader.py            # Data loading and processing
├── main.py                   # Training and evaluation script
└── requirements.txt          # Project dependencies
```

## Key Components

### Model Architecture (HotCakeModels.py)
- `HotCakeForSequenceClassification`: Main model class inheriting from BertPreTrainedModel
- Binary classification for fake news detection
- Utilizes BERT embeddings with custom classification layer
- Includes dropout for regularization

### Data Processing (data_loader.py)
- Custom dataset classes for news articles
- Stratified train-test split functionality
- Efficient batch processing
- Support for multiple datasets (PolitiFact, GossipCop, BuzzNews)

### Training Script (main.py)
- Configurable training parameters
- Support for CUDA acceleration
- TensorBoard integration
- Comprehensive evaluation metrics
- Learning rate scheduling with warmup

### Utilities
- Preprocessing utilities for text data
- Evaluation report generation
- Model checkpointing
- Performance analysis tools

## Requirements

- Python 3.7+
- PyTorch >= 1.7.0
- Transformers >= 4.5.0
- Additional dependencies in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WSDMS-EMNLP2023.git
cd WSDMS-EMNLP2023
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the model:
   - Edit configuration files in `config/` for your dataset
   - Adjust hyperparameters as needed

2. Prepare your data:
   - Place your dataset in the appropriate format
   - Use the provided preprocessing scripts

3. Train the model:
```bash
python main.py --cuda --enable_tensorboard --config_file PolitiFact.ini --outdir output
```

## Training

To train the model, use the following command:

```bash
python main.py --config_file config/PolitiFact.ini --outdir output --root sample_data
```

Key training parameters:
- `--num_train_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--train_batch_size`: Batch size for training (default: 8)
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation (default: 8)
- `--enable_tensorboard`: Enable TensorBoard logging
- `--patience`: Early stopping patience (default: 20)

The model will save the best checkpoint based on validation accuracy to the specified output directory.

### TensorBoard

To view training progress and metrics:
1. Enable TensorBoard logging with `--enable_tensorboard`
2. Run TensorBoard:
```bash
tensorboard --logdir runs
```

### Model Architecture

The model uses a BERT-based architecture (HotCakeForSequenceClassification) with the following components:
- BERT encoder for text representation
- Dropout layer for regularization
- Linear classifier for binary classification (real/fake news)

The model processes both the claim and evidence text, using BERT's sequence pair classification format.

## Model Configuration

Key parameters in configuration files:
- `bert_hidden_dim`: BERT hidden dimension size (default: 768)
- `train_batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `num_train_epochs`: Number of training epochs
- `max_len`: Maximum sequence length
- `gradient_accumulation_steps`: Steps for gradient accumulation

## Features

- BERT-based sequence classification
- Binary fake news detection
- TensorBoard integration for monitoring
- Configurable model architecture
- Comprehensive evaluation metrics
- Support for multiple datasets
- Efficient data processing pipeline

## Paper Overview

## WSDMS: Debunk Fake News via Weakly Supervised Detection of Misinforming Sentences

This repository implements the WSDMS (Weakly Supervised Detection of Misinforming Sentences) model described in our EMNLP 2023 paper. The model presents a novel approach to fake news detection by identifying sentence-level misinformation using weak supervision and contextual social media data.

### Problem Statement
Unlike conventional fake news detection methods that provide article-level verdicts, WSDMS aims to:
- Identify which specific sentences contain misinformation
- Leverage social context data for more nuanced detection
- Provide both sentence-level and article-level predictions

### Model Architecture

The WSDMS model consists of four key components:

1. **Input Embedding**
   - Uses Sentence-BERT (SBERT) for contextualized sentence representations
   - Encodes both news sentences and social media posts

2. **Sentence-Tree Linking**
   - Links article sentences to relevant social conversation trees
   - Utilizes cosine similarity and Kernel Graph Attention Networks (KGAT)
   - Creates relationships between sentences and social context

3. **Misinforming Sentence Detection**
   - Employs graph attention network for sentence-level analysis
   - Leverages social context from linked conversation trees
   - Computes misinformation probability for each sentence

4. **Article Veracity Prediction**
   - Applies weighted collective attention
   - Combines sentence-level predictions
   - Produces final article-level veracity prediction

### Datasets
The model has been tested on three datasets:
- **PolitiFact**: Political news fact-checking
- **GossipCop**: Entertainment news verification
- **BuzzNews**: Extended BuzzFeed news articles with Twitter conversations

Each dataset includes:
- News articles
- Associated social media conversations (Twitter)
- Ground truth labels

## Robustness Testing

We implement robustness testing based on the CheckList methodology (Ribeiro et al., 2020) to evaluate the model's behavior beyond accuracy metrics. See `tests/robustness_tests.py` for implementation details.

### Test Types
1. **Minimum Functionality Tests (MFT)**
   - Basic capabilities testing
   - Vocabulary and named entity handling
   - Simple negation understanding

2. **Invariance Tests (INV)**
   - Testing model stability
   - Location/name/number perturbations
   - Neutral phrase additions

3. **Directional Expectation Tests (DIR)**
   - Testing expected behavior changes
   - Negation handling
   - Strength intensifier effects

### Running Tests
```bash
python tests/robustness_tests.py --model_path /path/to/model --test_type all
```

## Running on Google Colab

Follow these steps to run the project on Google Colab:

1. Open a new Google Colab notebook
2. Clone the repository and install dependencies:
```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/WSDMS-EMNLP2023.git
%cd WSDMS-EMNLP2023

# Install required packages
!pip install -r requirements.txt

# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Mount Google Drive (optional, for saving model checkpoints):
```python
from google.colab import drive
drive.mount('/content/drive')

# Create output directory in Drive (optional)
!mkdir -p /content/drive/MyDrive/WSDMS-EMNLP2023/output
```

4. Download and prepare the dataset:
```python
# Create data directories
!mkdir -p sample_data/raw

# Download sample data (replace with your data source)
# !wget -O sample_data/raw/politifact_data.json YOUR_DATA_URL
```

5. Train the model:
```python
# For training with default parameters
!python main.py \
    --config_file config/PolitiFact.ini \
    --outdir /content/drive/MyDrive/WSDMS-EMNLP2023/output \
    --root sample_data \
    --enable_tensorboard

# For faster experimentation (reduced epochs)
!python main.py \
    --config_file config/PolitiFact.ini \
    --outdir /content/drive/MyDrive/WSDMS-EMNLP2023/output \
    --root sample_data \
    --num_train_epochs 10 \
    --enable_tensorboard
```

6. Monitor training with TensorBoard:
```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir runs
```

### Tips for Colab

1. Use a GPU runtime:
   - Go to Runtime → Change runtime type
   - Select GPU from Hardware accelerator dropdown

2. Prevent session timeouts:
   - Keep the browser tab active
   - Or use JavaScript to click the page periodically:
```javascript
function ClickConnect(){
    console.log("Working!");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

3. Save checkpoints to Google Drive:
   - Mount Drive as shown above
   - Use Drive path for `--outdir` to persist model checkpoints

4. Monitor GPU usage:
```python
# Monitor GPU usage
!nvidia-smi
```

5. For long training sessions:
   - Use smaller number of epochs for testing
   - Save checkpoints frequently
   - Consider using Colab Pro for longer runtimes

## License

[Add your license information here]

## Citation

[Add citation information for the research paper]

## Contributors

[Add contributor information]
