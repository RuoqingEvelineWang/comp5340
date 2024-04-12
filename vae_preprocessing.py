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

# Use only a subset due to ram constraint
num_rows = 10_000

# One-hot encode categorical features
ohe = OneHotEncoder()
encoded_cols = ohe.fit_transform(df[
            ['From Bank',
             'To Bank',
             'Receiving Currency',
             'Payment Currency',
             'Payment Format',
             'FromBankAcc',
             'ToBankAcc']][:num_rows])  # Use only a subset due to ram constraint
encoded_df = pd.DataFrame(encoded_cols.toarray(),
                          columns=ohe.get_feature_names_out(
                              ['From Bank',
                               'To Bank',
                               'Receiving Currency',
                               'Payment Currency',
                               'Payment Format',
                               'FromBankAcc',
                               'ToBankAcc']))

# Concat categorical features with other numerical features + labels
df = pd.concat([encoded_df, df[['Amount Paid', 'Amount Received', 'Is Laundering']][:num_rows]], axis=1)

# Save to disk
df.to_csv(f'data/HI_Small_Trans_onehot{str(num_rows)}.csv', index=False)
