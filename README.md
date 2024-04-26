## Transactions for Anti Money Laundering (AML)

### Set-up

1. If you are using Anaconda/Miniconda, create a conda environment using conda dependency file `pymc_env.yml`:

```python
conda env create -f pymc_env.yml
conda activate pymc_env
```

2. If you are using `pip` (recommended: Create virtual environment before `pip install`):

```python
pip install -r requirements. txt
```

### To update dependency files

1. Update `pymc_env.yml`:

```python
conda env export > pymc_env.yml
```

2. Update `requirements.txt` using Ananconda/Miniconda:

```python
conda list -e > requirements.txt
```

### Dataset Download

The dataset(s) used for this project can be downloaded from Kaggle using this [link](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

It is recommended to save the `csv` files into a folder named `data`.

### Processing Scripts

* `data_processing.ipynb`: Run standardised pre-processing steps on raw dataset.  
* `vae_preprocessing.py`: Model-specific preprocessing (additional) before training and inference for VAE.

## Train the Models and Make Predictions
We provide several Python files and notebooks in the root directoryfor each model type that we experimented with.
- `bayesian_logistic_regression.ipynb` Feature Engineering & Implementation of Bayesian Logistic Regression Models
- `naive_bayes.ipynb` Feature Engineering & Implementation of Naive Bayes model
- `gmm.ipynb` Feature Engineering & Implementation of Gaussian Mixture Model. Dual GMMs is the model \#3 in the notebook
- `vae_train.py` Training of VAE model
- `vae_inference.ipynb` Prediction results and metrics generation, and latent space visualization of VAE 
