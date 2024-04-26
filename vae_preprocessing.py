import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Preprocess train
df_train = pd.read_csv("./data/HI-Small_Trans_processed_w_original_train.csv")
df_test = pd.read_csv("./data/HI-Small_Trans_processed_w_original_test.csv")

# https://finance.yahoo.com/quote/BTC-USD/history
# https://www.exchangerates.org.uk/US-Dollar-USD-currency-table.html
#
# use the average ex rates over Sep 2022 per Timestamp
#
ex_rates = {
    'Australian Dollar': 1.4958,
    'Bitcoin': 1/20201.19,
    'Brazil Real': 5.2299,
    'Canadian Dollar': 1.3316,
    'Euro': 1.009, 
    'Mexican Peso': 20.055,
    'Ruble': 60.0435,
    'Rupee': 80.2343,
    'Saudi Riyal': 3.7598,
    'Shekel': 3.4489,
    'Swiss Franc': 0.973,
    'UK Pound': 0.836,
    'US Dollar': 1.0,
    'Yen': 143.0069,
    'Yuan': 7.0144,
}

df_train['AmountReceivedUSD'] = df_train['AmountReceived']
df_train['AmountPaidUSD'] = df_train['AmountPaid']
df_test['AmountReceivedUSD'] = df_test['AmountReceived']
df_test['AmountPaidUSD'] = df_test['AmountPaid']
for currency, rate in ex_rates.items():
    mask = df_train['Receiving Currency'] == currency
    df_train.loc[mask, 'AmountReceivedUSD'] /= rate
    mask = df_train['Payment Currency'] == currency
    df_train.loc[mask, 'AmountPaidUSD'] /= rate
    #
    mask = df_test['Receiving Currency'] == currency
    df_test.loc[mask, 'AmountReceivedUSD'] /= rate
    mask = df_test['Payment Currency'] == currency
    df_test.loc[mask, 'AmountPaidUSD'] /= rate

# Keep only selected columns
df_train = df_train[["FromAccount", "ToAccount", \
                    "PaymentFormat", "AmountReceivedUSD", "AmountPaidUSD", \
                    "Is Laundering"]]

df_test = df_test[["FromAccount", "ToAccount", \
                    "PaymentFormat", "AmountReceivedUSD", "AmountPaidUSD", \
                    "Is Laundering"]]

scaler = StandardScaler()

tempX = df_train.drop(columns=["Is Laundering"])
tempX = scaler.set_output(transform='pandas').fit_transform(tempX)
df_train = pd.concat([tempX, df_train["Is Laundering"]], axis=1)

tempX = df_test.drop(columns=["Is Laundering"])
tempX = scaler.set_output(transform='pandas').transform(tempX)
df_test = pd.concat([tempX, df_test["Is Laundering"]], axis=1)

# Save to disk
df_train.to_csv(f'./data/HI_Small_Trans_ordinal_train_vae-.csv', index=False)
df_test.to_csv(f'./data/HI_Small_Trans_ordinal_test_vae-.csv', index=False)
