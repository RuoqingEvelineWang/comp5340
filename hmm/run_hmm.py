import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from hmmlearn import hmm
import random
import warnings
import json
import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import multiprocessing as mp
warnings.filterwarnings('ignore')
result_queue = mp.Queue()

def write_to_file(result_path):
    global result_queue
    cnt = 1
    result = open(result_path, 'w')
    while True:
        line = result_queue.get()
        if line != None:
            result.write(json.dumps(line) + '\n')
            cnt = cnt + 1
        else:
            break
        if cnt % 100 == 0:
            print('processed {}'.format(cnt))
    result.close()
    print('Processed done. {}'.format(cnt))

def train_test_seq(k):
    try:
        tmp_train_seq = train_grouped_data[k]
        tmp_test_seq = test_grouped_data[k]
        train_len = tmp_train_seq.shape[0]
        test_len = tmp_test_seq.shape[0]
        input_data = pd.concat((tmp_train_seq, tmp_test_seq))
        processed_input_data, labels = preprocessing(input_data)
        train_input = processed_input_data[:train_len, :]
        # test_input = processed_input_data[train_len:, :]
        # print(train_input.shape, test_input.shape)
        try:
            model = hmm.GMMHMM(n_components=2, n_mix=2, init_params='mcw', n_iter=100)
            model.transmat_ = np.array([np.random.dirichlet([0.999, 0.001]),
                                        np.random.dirichlet([0.999, 0.001])])
            model.startprob_ = np.array([1.0, 0.0])
            model.fit(train_input, train_len)
            scores = [model.score(train_input)]
            # for i in range(test_len):
                # new_seq = np.vstack((train_input[i+1:, :], test_input[:i+1, :]))
            new_seq = processed_input_data[test_len:, :]
            # print(new_seq.shape)
            scores.append(model.score(new_seq))
            pred_z = model.predict_proba(new_seq)
            # result_queue.put({'idx': k, 'score': scores, 'prob': pred_z})
            return {'idx': k, 'valid': True, 'label': labels[-test_len:, :].tolist(), 'score': scores, 'prob': pred_z.tolist()}
        except:
            return {'idx': k, 'valid': False, 'label': labels[-test_len:, :].tolist()}
    except:
        # time.sleep(1)
        return {'idx': k, 'valid': False}

def group_by_accounts(df, col1, col2, other_columns):
    # Reshape the DataFrame to have a single account column, while preserving other specified columns
    melted_df = pd.melt(df.reset_index(), id_vars=['index'] + other_columns, value_vars=[col1, col2])
    melted_df.rename(columns={'value': 'Account'}, inplace=True)
    
    # Group by account and collect all unique indices for each account
    account_indices = melted_df.groupby('Account')['index'].unique()
    
    # Create a DataFrame for each account group using the collected indices
    account_group_dataframes = {account: df.loc[indices].drop_duplicates() for account, indices in account_indices.items()}
    return account_group_dataframes

def preprocessing(df):
    encode_curr = LabelEncoder().fit(pd.concat([df['ReceivingCurrency'], df['PaymentCurrency']], ignore_index=True)) # For all Currency 
    encode_paym_format = LabelEncoder().fit(df['PaymentFormat']) # Payment Format
    encode_acct = LabelEncoder().fit(pd.concat([df['FromAccount'], df['ToAccount']], ignore_index=True)) # For all unique Account
    encode_bank = LabelEncoder().fit(pd.concat([df['FromBank'], df['ToBank']], ignore_index=True)) # For all unique Bank codes
    scaler = MinMaxScaler()
    # clean_df = df[['FromAccount', 'ToAccount', 'FromBank', 'ToBank', 'ReceivingCurrency', 'PaymentCurrency', 'PaymentFormat', 'AmountPaid', 'AmountReceived']]
    clean_df = df[['FromAccount', 'ToAccount', 'ReceivingCurrency', 'PaymentCurrency', 'PaymentFormat', 'AmountPaid', 'AmountReceived']]
    clean_df['FromAccount'] = encode_acct.transform(df['FromAccount'])
    clean_df['ToAccount'] = encode_acct.transform(df['ToAccount'])
    # clean_df['FromBank'] = encode_bank.transform(df['FromBank'])
    # clean_df['ToBank'] = encode_bank.transform(df['ToBank'])
    clean_df['ReceivingCurrency'] = encode_curr.transform(df['ReceivingCurrency'])
    clean_df['PaymentCurrency'] = encode_curr.transform(df['PaymentCurrency'])
    clean_df['PaymentFormat'] = encode_paym_format.transform(df['PaymentFormat'])
    clean_df[['AmountPaid', 'AmountReceived']] = scaler.fit_transform(clean_df[['AmountPaid', 'AmountReceived']])

    labels = df[['Is Laundering']]
    return clean_df.to_numpy(), labels.to_numpy()


if __name__ == '__main__':
    train_df = pd.read_csv('HI-Small_Trans_processed_w_original_train.csv')
    test_df = pd.read_csv('HI-Small_Trans_processed_w_original_test.csv')

    clean_train_df = train_df.drop(["From Bank", "To Bank", "Account", "Account.1", "Receiving Currency", "Payment Currency", "Payment Format"], axis=1)
    clean_test_df = test_df.drop(["From Bank", "To Bank", "Account", "Account.1", "Receiving Currency", "Payment Currency", "Payment Format"], axis=1)
    print('train data: {}'.format(clean_train_df.shape))
    print('test data: {}'.format(clean_test_df.shape))

    other_columns = [x for x in clean_train_df.columns if x not in ['FromAccount', 'ToAccount']]
    train_grouped_data = group_by_accounts(clean_train_df, 'FromAccount', 'ToAccount', other_columns)
    test_grouped_data = group_by_accounts(clean_test_df, 'FromAccount', 'ToAccount', other_columns)
    print('train seq number: {}'.format(len(train_grouped_data)))
    print('test seq number: {}'.format(len(test_grouped_data)))

    data_list = []
    counter = 0
    for k, v in test_grouped_data.items():
        if k not in train_grouped_data:
            continue
        data_list.append(k)
        counter += 1
        if counter > 10:
            break
    print(data_list)

    num_processes = 30
    output_file = 'output_results.json'

    # write_p = mp.Process(target=write_to_file, args=(output_file,))
    # write_p.start()
    # p = mp.Pool(num_processes)
    # p.map(train_test_seq, data_list)
    # p.close()
    # p.join()
    # result_queue.put(None)
    # write_p.join()

    out = open(output_file, 'w')
    for k in tqdm.tqdm(test_grouped_data):
        start = time.time()
        out_info = train_test_seq(k)
        if out_info is None:
            continue
        out.write(json.dumps(out_info) + '\n')
        # end = time.time()
        # print(end - start)
    out.close()