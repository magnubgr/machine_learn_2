import numpy as np
import pandas as pd

file = "../data/default_credit_card_data.xls"

from pandas import ExcelWriter
from pandas import ExcelFile

# df = pd.read_excel(file, sheetname='Data')
# df = pd.read_excel(file, header=[0, 1], sheetname="Data")
# df = pd.read_excel(file, header=[0, 1], sheetname="Data")
df = pd.read_excel(file, header=1, skiprows=0, index_col=0)

df.rename(
    index=str,
    columns={"default payment next month": "defaultPaymentNextMonth"},
    inplace=True
)

print(df)
print("Column headings:")
columns = df.columns
print(columns)

id_labels = df.index

# limit_bal = df['X1']
# sex = df['X2']
# education = df['X3']
# marriage = df['X4']
# age = df['X5']
# pay_0 = df['X6']
# pay_2 = df['X7']
# pay_3 = df['X8']
# pay_4 = df['X9']
# pay_5 = df['X10']
# pay_6 = df['X11']
# bill_amt1 = df['X12']
# bill_amt2 = df['X13']
# bill_amt3 = df['X14']
# bill_amt4 = df['X15']
# bill_amt5 = df['X16']
# bill_amt6 = df['X17']
# pay_amt1 = df['X18']
# pay_amt2 = df['X19']
# pay_amt3 = df['X20']
# pay_amt4 = df['X21']
# pay_amt5 = df['X22']
# pay_amt6 = df['X23']
# default = df['Y']


# print(df)
