# Training and model code adapted from https://github.com/Michedev/VAE_anomaly_detection

import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import yaml
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from model.VAE import VAEAnomalyTabular
from vae_dataset import rand_dataset, VAEDataset

from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'

def make_folder_run() -> Path:
    """
    Get the folder where to store the experiment. 
    The folder is named with the current date and time.
    
    Returns:
        Path: the path to the folder where to store the experiment
    """
    checkpoint_folder = SAVED_MODELS / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_folder.makedirs_p()
    return checkpoint_folder


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', '-i', type=int, required=True, dest='input_size', help='Number of input features. In 1D case it is the vector length, in 2D case it is the number of channels')
    parser.add_argument('--latent-size', '-l', type=int, required=True, dest='latent_size', help='Size of the latent space')
    parser.add_argument('--hidden-1', '-h1', type=int, required=False, dest='hidden_1', help='Size of hidden layer 1 in encoder and decoder')
    parser.add_argument('--hidden-2', '-h2', type=int, required=False, dest='hidden_2', help='Size of hidden layer 2 in encoder and decoder')
    parser.add_argument('--num-resamples', '-L', type=int, dest='num_resamples', default=10,
                        help='Number of resamples in the latent distribution during training')
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=100, help='Number of epochs to train for')
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=32)
    parser.add_argument('--device', '-d', '--accelerator', type=str, dest='device', default='gpu', help='Device to use for training. Can be cpu, gpu or tpu', choices=['cpu', 'gpu', 'tpu'])
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3, help='Learning rate')
    parser.add_argument('--no-progress-bar', action='store_true', dest='no_progress_bar')
    parser.add_argument('--steps-log-loss', type=int, dest='steps_log_loss', default=1_000, help='Number of steps between each loss logging')
    parser.add_argument('--steps-log-norm-params', type=int, 
                        dest='steps_log_norm_params', default=1_000, help='Number of steps between each model parameters logging')

    return parser.parse_args()


def main():
    """
    Main function to train the VAE model
    """
    args = get_args()
    print(args)
    experiment_folder = make_folder_run()

    # copy model folder into experiment folder
    ROOT.joinpath('model').copytree(experiment_folder / 'model')

    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    model = VAEAnomalyTabular(args.input_size, args.latent_size, args.hidden_1, args.hidden_2, args.num_resamples, lr=args.lr)

    df_train = pd.read_csv('data/HI_Small_Trans_ordinal_train_vae-.csv')

    # Split into train and val sets
    df_train = df_train.drop(columns=['Is Laundering'])  #.sample(50_000, random_state=0)

    ratio = 0.89
    train_set, val_set = train_test_split(df_train, test_size=ratio, random_state=0)
    val_ratio = 0.1
    train_set, val_set = train_test_split(train_set, test_size=val_ratio, random_state=0)

    # val_ratio = 0.1
    # train_set, val_set = train_test_split(df_train, test_size=val_ratio, random_state=0)
    train_set = VAEDataset(train_set.reset_index(drop=True))
    val_set = VAEDataset(val_set.reset_index(drop=True))

    train_dloader = DataLoader(train_set, args.batch_size, shuffle=True)

    val_dloader = DataLoader(val_set, args.batch_size, shuffle=False)

    checkpoint = ModelCheckpoint(
        filename=experiment_folder / '{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    
    trainer = Trainer(accelerator='auto', max_epochs=args.epochs, callbacks=[checkpoint],)
    trainer.fit(model, train_dloader, val_dloader)

if __name__ == '__main__':
    main()