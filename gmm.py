# import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, LabelEncoder, OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_transactions(df):
    n_samples = 5000

    # Separate data into subsets based on classes
    class_a = df[df['IsLaundering'] == 0].sample(n_samples)
    class_b = df[df['IsLaundering'] == 1]

    print("Class A:", len(class_a))
    print("Class B:", len(class_b))

    # Plot histograms for each class
    sns.histplot(data=class_a, x='AmountReceivedUSD', color='blue', label='Class A', kde=True)
    sns.histplot(data=class_b, x='AmountReceivedUSD', color='red', label='Class B', kde=True)

    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title('Distribution of Transaction Amounts per Class')
    plt.legend()
    plt.show()


plot_transactions(clean_df)
