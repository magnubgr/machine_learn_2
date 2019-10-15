import numpy as np
import pandas as pd


def read_credit_card_file():

    file = "../data/default_credit_card_data.xls"

    # df = pd.read_excel(file, header=[0, 1], sheetname="Data")
    df = pd.read_excel(file, header=1, skiprows=0, index_col=0)

    df.rename(
        index=str,
        columns={"default payment next month": "defaultPaymentNextMonth"},
        inplace=True
    )

    columns_df = df.columns
    columns_np = columns_df.values

    id_labels_df = df.index
    id_labels_np = id_labels_df.values

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    # print(X)
    # print(y)

    return X, y

    """ This is in case we need the individual columns
    limit_bal = df['LIMIT_BAL']
    sex = df['SEX']
    education = df['EDUCATION']
    marriage = df['MARRIAGE']
    age = df['AGE']
    pay_0 = df['PAY_0']
    pay_2 = df['PAY_2']
    pay_3 = df['PAY_3']
    pay_4 = df['PAY_4']
    pay_5 = df['PAY_5']
    pay_6 = df['PAY_6']
    bill_amt1 = df['BILL_AMT1']
    bill_amt2 = df['BILL_AMT2']
    bill_amt3 = df['BILL_AMT3']
    bill_amt4 = df['BILL_AMT4']
    bill_amt5 = df['BILL_AMT5']
    bill_amt6 = df['BILL_AMT6']
    pay_amt1 = df['PAY_AMT1']
    pay_amt2 = df['PAY_AMT2']
    pay_amt3 = df['PAY_AMT3']
    pay_amt4 = df['PAY_AMT4']
    pay_amt5 = df['PAY_AMT5']
    pay_amt6 = df['PAY_AMT6']
    default = df['defaultPaymentNextMonth']
    """

    # print(df)

if __name__ == "__main__":
    read_credit_card_file()
