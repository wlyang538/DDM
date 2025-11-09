# Defensive Dual Masking for Robust Adversarial Defense



This repository provides an implementation of **Defensive Dual Masking (DDM)**, as proposed in the paper
 *“Defensive Dual Masking for Robust Adversarial Defense.”*

The project supports **offline evaluation** of adversarial robustness by training a DDM classifier, loading precomputed adversarial samples, and testing defense effectiveness with and without inference-time masking.

------



## Repository Structure

```
.
├── config.py                  # Configuration file (paths, hyperparameters, etc.)
├── train.py                   # Model training script
├── attack.py                  # Evaluation & adversarial loading script
├── requirements.txt
├── bert-base-uncased/  			 # Local pretrained BERT model
├── main.py
└── data/
    ├── sst2/
    ├── adv/								   # Precomputed adversarial examples (.json)
    └── .../                   # Other datasets like mr etc.
```

------

## Environment Setup

It is recommended to use **conda** to manage dependencies.

```
conda create -n ddm python=3.10 -y
conda activate ddm
```

Then install dependencies:

```
pip install -r requirements.txt
```

> **Note:**
>  The project is designed for **offline environments**.
>  Please ensure the pretrained model `bert-base-uncased/` and datasets are stored locally as indicated in `config.py`.

------

##  Running the Pipeline

### 1. Train the Model

Before any adversarial evaluation, train the DDM model on your chosen dataset:

```
python train.py --dataset sst2 --epochs 10
```

This will produce:

- Model checkpoints under `outputs/best_model`
- Token frequency file `token_freq.json`

------

### 2. Evaluate with Precomputed Adversarial Samples

After training, you can evaluate the model’s robustness using adversarial data.
 For example, with the **TextFooler** attack on **SST-2**:

```
python attack.py \
  --attack_mode textfooler \
  --dataset sst2 \
  --limit 1000 \					# How many samples you want to test.
  --batch_size 16 \
  --replace_ratio 0.2 \		# Mask ratio when infer
  --test									# Use test split
```

This will:

- Load the pretrained DDM model
- Evaluate accuracy on clean and adversarial data
- Measure the **attack success rate** (SUCC)
- Report performance with and without DDM masking

------

## Precomputed Adversarial Samples

Because this code is designed for **offline environments**, adversarial examples must be **generated beforehand** using your chosen attack method.

### File naming convention

Place the adversarial files under `data/adv/` and name them as:

```
{dataset}_{attack_mode}_adv.json
```

For example:

- `sst2_textfooler_adv.json`
- `mr_deepwordbug_adv.json`

The program will automatically detect and load the correct file.

### Expected JSON format

Each file should contain a list of dictionaries:

```
[
  {
    "label": 1,
    "text": "it 's a tempt nad often affecting journey ."
  },
  {
    "label": 0,
    "text": "unflinchinglyy raw and heroic"
  }
]
```

### Notes

- `label` must be an integer (e.g., 0 or 1)
- `text` is the adversarial sentence string
- If you use your own dataset, ensure paths are correctly set in `config.py`

------

## Configuration

Most parameters are controlled via **`config.py`**, including:

- Model architecture and hyperparameters
- Paths to datasets (e.g., `Config.sst2_path`, `Config.mr_path`)
- Output directory (`Config.output_dir`)
- Inference masking ratio (`Config.suspicious_ratio`)
- Path to the pretrained BERT model (`bert-base-uncased/`)

------

## Output

During evaluation, logs and results will include:

- Clean accuracy (CLA)
- Adversarial accuracy (CAA)
- Attack success rate (SUCC)
- Optional file outputs, e.g. masked text samples

Example snippet to save masked results:

```
with open("sample_after_mask.txt", "a", encoding="utf-8") as f:
    f.write(f"[Sample {b}] After masking:\n{text_after_mask}\n\n")
    
    
# Example output :
[Sample 0] After masking:
[CLS] it's a te [MASK] [MASK] often affecting journey. [SEP]

[Sample 1] After masking:
[CLS] unflinchinglyy raw and [MASK] [SEP]
```

------

## 