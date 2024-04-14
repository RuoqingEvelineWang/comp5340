import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv('data/HI-Small_Trans.csv')

# Concat Bank ID with Account ID (in case of Account ID duplicates)
df['FromBankAcc'] = df.iloc[:,1].astype(str) + '_' + df.iloc[:,2]
df['ToBankAcc'] = df.iloc[:,3].astype(str) + '_' + df.iloc[:,4]

df = df.drop(columns=['Timestamp', 'Account', 'Account.1'])

# Convert data types to categorical where applicable
df['From Bank'] = df['From Bank'].astype('category')
df['To Bank'] = df['To Bank'].astype('category')
df['Receiving Currency'] = df['Receiving Currency'].astype('category')
df['Payment Currency'] = df['Payment Currency'].astype('category')
df['Payment Format'] = df['Payment Format'].astype('category')
df['Is Laundering'] = df['Is Laundering'].astype('category')
df['FromBankAcc'] = df['FromBankAcc'].astype('category')
df['ToBankAcc'] = df['ToBankAcc'].astype('category')

# # Use only a subset due to ram constraint
# num_rows = 11_000
# num_ml = 1_000

# df_10k = df[:num_rows]
# df_10k = df_10k[df_10k['Is Laundering'] == 0]
# print(f"df_10k{df_10k.shape}")

# df_ml = df[df['Is Laundering'] == 1][:num_ml] 

# df_10k_ml = pd.concat([df_10k.reset_index(drop=True), df_ml.reset_index(drop=True)], axis=0)
# print(f"df_10k_ml{df_10k_ml.shape}")

# Ordinal encode categorical features
oe = OrdinalEncoder()
encoded_cols = oe.fit_transform(df[
            ['From Bank',
             'To Bank',
             'Receiving Currency',
             'Payment Currency',
             'Payment Format',
             'FromBankAcc',
             'ToBankAcc']])  # Use only a subset due to ram constraint
encoded_df = pd.DataFrame(encoded_cols)  #.toarray(),
                        #   columns=ohe.get_feature_names_out(
                        #       ['From Bank',
                        #        'To Bank',
                        #        'Receiving Currency',
                        #        'Payment Currency',
                        #        'Payment Format',
                        #        'FromBankAcc',
                        #        'ToBankAcc']))
# print(f"encoded_df{encoded_df.shape}")
# df_others = df[['Amount Paid', 'Amount Received', 'Is Laundering']][:num_rows]
# df_others = df_others[df_others['Is Laundering'] == 0]
# print(f"df_others{df_others.shape}")
# df_ml_others = df_ml[['Amount Paid', 'Amount Received', 'Is Laundering']]
# print(f"df_ml_others{df_ml_others.shape}")
# df_second = pd.concat([df_others.reset_index(drop=True), df_ml_others.reset_index(drop=True)], axis=0)
# print(f"df_second{df_second.shape}")

df_others = df[['Amount Paid', 'Amount Received', 'Is Laundering']]
# Concat categorical features with other numerical features + labels
df = pd.concat([encoded_df, df_others], axis=1)

# Save to disk
df.to_csv(f'data/HI_Small_Trans_ordinal.csv', index=False)
