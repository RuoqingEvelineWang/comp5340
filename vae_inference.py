import argparse
import torch
from path import Path
from torch.utils.data import DataLoader
from model.VAE import VAEAnomalyTabular
from vae_dataset import VAEDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, LabelEncoder, OneHotEncoder, StandardScaler


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

    args = get_args()
    print(args)

    dataset = pd.read_csv('data/HI_Small_Trans_ordinal.csv')

    scaler = StandardScaler()
    temp = dataset.drop(columns=['Is Laundering'])
    temp = scaler.set_output(transform='pandas').fit_transform(temp)
    dataset = pd.concat([temp, dataset['Is Laundering']], axis=1)

    # work on a subset of 25,000 class 0 and ~5,000 class 1
    temp = dataset[dataset['Is Laundering'] == 0].sample(25_000, random_state=0)
    # first separate out test set
    class1 = dataset[dataset['Is Laundering']==1]
    dataset = pd.concat([temp, class1], axis=0)
    num_sample = len(class1)
    test_ratio = round(num_sample / len(dataset[dataset['Is Laundering'] == 0]), 4)
    dataset = dataset[dataset['Is Laundering'] == 0]
    # test set has only class0
    dataset, test_set = train_test_split(dataset, test_size=test_ratio, random_state=0)
    test_set = VAEDataset(test_set.drop(columns=['Is Laundering']).reset_index(drop=True))
    test_dloader = DataLoader(test_set, args.batch_size, shuffle=False)
    
    # class1 has all class1 rows
    class1 = VAEDataset(class1.drop(columns=['Is Laundering']).reset_index(drop=True))
    class1_dloader = DataLoader(class1, args.batch_size, shuffle=False)

    checkpointfile = torch.load(
        'C:\\YKT\\NUS_MComp\\CS5340_project\\comp5340\\lightning_logs\\version_49\\checkpoints\\last.ckpt')
    net = VAEAnomalyTabular(args.input_size, args.latent_size, args.hidden_1, args.hidden_2, args.num_resamples, lr=args.lr)
    net.load_state_dict(checkpointfile["state_dict"])
    net.eval()

    for batch in test_dloader:
        print("Test set (class 0)")
        with torch.no_grad():
            print(net.is_anomaly(batch))
    
    for batch in class1_dloader:
        print("Test set (class 1)")
        with torch.no_grad():
            print(net.is_anomaly(batch))

if __name__ == '__main__':
    main()