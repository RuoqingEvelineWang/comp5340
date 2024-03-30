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