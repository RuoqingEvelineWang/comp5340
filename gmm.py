import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/HI-Small_Trans.csv')

#
# https://finance.yahoo.com/quote/BTC-USD/history
# https://www.exchangerates.org.uk/US-Dollar-USD-currency-table.html
#
# use the average ex rates over Sep 2022 per Timestamp
#
ex_rates = {
    'US Dollar': 1.0,
    'Bitcoin': 1/20201.19,
    'Euro': 1.009,
    'Australian Dollar': 1.4958,
    'Yuan': 7.0144,
    'Rupee': 80.2343,
    'Mexican Peso': 20.055,
    'Yen': 143.0069,
    'UK Pound': 0.836,
    'Ruble': 60.0435,
    'Canadian Dollar': 1.3316,
    'Swiss Franc': 0.973,
    'Brazil Real': 5.2299,
    'Saudi Riyal': 3.7598,
    'Shekel': 3.4489,
}

df['AmountReceivedUSD'] = df['Amount Received']
df['AmountPaidUSD'] = df['Amount Paid']
for currency, rate in ex_rates.items():
    print(f"{currency}: {rate}")
    mask = df['Receiving Currency'] == currency
    df.loc[mask, 'AmountReceivedUSD'] /= rate
    mask = df['Payment Currency'] == currency
    df.loc[mask, 'AmountPaidUSD'] /= rate

df['FromBankAcc'] = df.iloc[:, 1].astype(str) + '_' + df.iloc[:, 2]
df['ToBankAcc'] = df.iloc[:, 3].astype(str) + '_' + df.iloc[:, 4]

# Nominal Encoding
encode_curr = LabelEncoder().fit(
    pd.concat([df['Receiving Currency'], df['Payment Currency']], ignore_index=True))  # For all Currency
encode_paym_format = LabelEncoder().fit(df['Payment Format'])  # Payment Format
encode_acct = LabelEncoder().fit(
    pd.concat([df['FromBankAcc'], df['ToBankAcc']], ignore_index=True))  # For all unique Account
encode_bank = LabelEncoder().fit(
    pd.concat([df['From Bank'], df['To Bank']], ignore_index=True))  # For all unique Bank codes

clean_df = pd.DataFrame()
clean_df['Timestamp'] = pd.to_datetime(df['Timestamp'])
clean_df['FromAccount'] = encode_acct.transform(df['FromBankAcc'])
clean_df['ToAccount'] = encode_acct.transform(df['ToBankAcc'])
clean_df['FromBank'] = encode_bank.transform(df['From Bank'])
clean_df['ToBank'] = encode_bank.transform(df['To Bank'])
clean_df['ReceivingCurrency'] = encode_curr.transform(df['Receiving Currency'])
clean_df['PaymentCurrency'] = encode_curr.transform(df['Payment Currency'])
clean_df['PaymentFormat'] = encode_paym_format.transform(df['Payment Format'])
# clean_df['AmountPaid'] = df['Amount Paid']
# clean_df['AmountReceived'] = df['Amount Received']
clean_df['IsLaundering'] = df['Is Laundering']
clean_df['AmountReceivedUSD'] = df['AmountReceivedUSD']
clean_df['AmountPaidUSD'] = df['AmountPaidUSD']

class_0 = clean_df[clean_df['IsLaundering'] == 0]
class_1 = clean_df[clean_df['IsLaundering'] == 1]
print("Class 0:", len(class_0))
print("Class 1:", len(class_1))

# Find the minimum number of samples among all classes
min_samples = clean_df.groupby('IsLaundering').size().min()
n_samples = {
    0: min_samples * 1,
    1: min_samples,
}

# Sample an unequal number of samples from each class
class_0 = clean_df[clean_df['IsLaundering'] == 0].sample(n_samples[0], replace=False)  # , random_state=42)
class_1 = clean_df[clean_df['IsLaundering'] == 1].sample(n_samples[1], replace=False)

sampled_df = pd.concat([class_0, class_1])

# sampled_df now contains an equal number of samples from each class
Z = sampled_df[['FromAccount', 'ToAccount', 'FromBank', 'ToBank', 'ReceivingCurrency', 'PaymentCurrency', 'PaymentFormat', 'AmountPaidUSD', 'AmountReceivedUSD']]
scaler = StandardScaler()
X = scaler.set_output(transform='pandas').fit_transform(Z)
y = sampled_df[['IsLaundering']]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # , random_state=42)

# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_resampled, y_resampled)  # X_train)

# Predict posterior probabilities for test data
probs = gmm.predict_proba(X_test)  # Probabilities of each sample belonging to each component
predictions = np.argmax(probs, axis=-1)

print(X_test.shape)
print(probs.shape)
print(predictions.shape)

metrics = {
    'accuracy': accuracy_score(y_test, predictions),
    'f1_score': f1_score(y_test, predictions, average='weighted'),  # Or use 'macro', 'micro', or None for different averaging methods
    'precision': precision_score(y_test, predictions),
    'recall': recall_score(y_test, predictions),
}
print("Metrics:", metrics)