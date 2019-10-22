import numpy as np
import pandas as pd
import sklearn

## Also xlrd is needed

def read_credit_card_file():

    file = "../data/default_credit_card_data.xls" # fix this pointer

    # df = pd.read_excel(file, header=[0, 1], sheetname="Data")
    df = pd.read_excel(file, header=1, skiprows=0, index_col=0)

    df.rename(
        index=str,
        columns={"default payment next month": "default"},
        inplace=True
    )

    from IPython.display import display
    display(df)


    ### Check if df["SEX"] has other values than 1,2
    print( np.unique( df["SEX"].values) )
    ### Check if df["EDUCATION"] has other values than 1,2,3,4 ---- apparently 0,5,6
    print( np.unique( df["EDUCATION"].values) )
    ### Check if df["MARRIAGE"] has other values than 1,2,3    ---- apparently 0
    print( np.unique( df["MARRIAGE"].values) )

    ## Consider one-hot encoding PAY_0-PAY_6... but lots of work
    ### What does 0 mean?
    #### Test out using just PAYed on time or not (0, 1)

    # One-hot encoding the gender
    df["MALE"] = (df["SEX"]==1).astype("int")
    df.drop("SEX", axis=1, inplace=True)

    # One-hot encoding for education
    df["GRADUATE_SCHOOL"] = (df["EDUCATION"]==1).astype("int")
    df["UNIVERSITY"] = (df["EDUCATION"]==2).astype("int")
    df["HIGH_SCHOOL"] = (df["EDUCATION"]==3).astype("int")
    df.drop("EDUCATION", axis=1, inplace=True)

    # One-hot encoding for marriage
    df["MARRIED"] = (df["MARRIAGE"]==1).astype("int")
    df["SINGLE"] = (df["MARRIAGE"]==2).astype("int")
    df.drop("MARRIAGE", axis=1, inplace=True)

    # display(df)



    # columns_df = df.columns
    # columns_np = columns_df.values

    # id_labels_df = df.index
    # id_labels_np = id_labels_df.values

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    # print(np.shape(X))
    # print(np.shape(y))

    return X, y

'''
    This is in case we need the individual columns
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
'''

    # print(df)

if __name__ == "__main__":
    read_credit_card_file()
