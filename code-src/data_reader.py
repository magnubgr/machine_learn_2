import numpy as np
import pandas as pd
import sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler #, OneHotEncoder
from sklearn import metrics


## Also xlrd is needed

def read_credit_card_file():

    path = os.path.dirname(os.path.realpath(__file__))
    file = path + "/../data/default_credit_card_data.xls" # fix this pointer

    # df = pd.read_excel(file, header=[0, 1], sheetname="Data")
    df = pd.read_excel(file, header=1, skiprows=0, index_col=0)

    df.rename(
        index=str,
        columns={"default payment next month": "DEFAULT"},
        inplace=True
    )

    print(df)


    ### Check if df["SEX"] has other values than 1,2
    print("SEX", np.unique( df["SEX"].values) )
    ### Check if df["EDUCATION"] has other values than 1,2,3,4 ---- apparently 0,5,6
    print("EDUCATION", np.unique( df["EDUCATION"].values) )
    ### Check if df["MARRIAGE"] has other values than 1,2,3    ---- apparently 0
    print("MARRIAGE", np.unique( df["MARRIAGE"].values) )
    ### Check if df["PAY_0"] has other values than -1,..,9
    print("PAY_0", np.unique( df["PAY_0"].values) )
    ### Check if df["PAY_2"] has other values than -1,..,9
    print("PAY_2", np.unique( df["PAY_2"].values) )
    ### Check if df["PAY_3"] has other values than -1,..,9
    print("PAY_3", np.unique( df["PAY_3"].values) )
    ### Check if df["PAY_4"] has other values than -1,..,9
    print("PAY_4", np.unique( df["PAY_4"].values) )
    ### Check if df["PAY_5"] has other values than -1,..,9
    print("PAY_5", np.unique( df["PAY_5"].values) )
    ### Check if df["PAY_6"] has other values than -1,..,9
    print("PAY_6", np.unique( df["PAY_6"].values) )


    ## Drop rows that are outside of features given
    df = df[(df.EDUCATION != 0) & 
            (df.EDUCATION != 5) & 
            (df.EDUCATION != 6)]
    df = df[ (df.MARRIAGE != 0) ]

    for dfpay in [df.PAY_0, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]:
        df = df[(dfpay != -2) ]
                # &(dfpay != 0)]

    ### Check if df["SEX"] has other values than 1,2
    print("SEX", np.unique( df["SEX"].values) )
    ### Check if df["EDUCATION"] has other values than 1,2,3,4 ---- apparently 0,5,6
    print("EDUCATION", np.unique( df["EDUCATION"].values) )
    ### Check if df["MARRIAGE"] has other values than 1,2,3    ---- apparently 0
    print("MARRIAGE", np.unique( df["MARRIAGE"].values) )
    ### Check if df["PAY_0"] has other values than -1,..,9
    print("PAY_0", np.unique( df["PAY_0"].values) )
    ### Check if df["PAY_2"] has other values than -1,..,9
    print("PAY_2", np.unique( df["PAY_2"].values) )
    ### Check if df["PAY_3"] has other values than -1,..,9
    print("PAY_3", np.unique( df["PAY_3"].values) )
    ### Check if df["PAY_4"] has other values than -1,..,9
    print("PAY_4", np.unique( df["PAY_4"].values) )
    ### Check if df["PAY_5"] has other values than -1,..,9
    print("PAY_5", np.unique( df["PAY_5"].values) )
    ### Check if df["PAY_6"] has other values than -1,..,9
    print("PAY_6", np.unique( df["PAY_6"].values) )

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


    # df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    # df.drop_duplicates()

    print(df)



    # columns_df = df.columns
    # columns_np = columns_df.values

    # id_labels_df = df.index
    # id_labels_np = id_labels_df.values


    X = df.loc[:, df.columns != 'DEFAULT'].values
    y = df.loc[:, df.columns == 'DEFAULT'].values

    ## Scale the features. So that for example LIMIT_BAL isnt larger than AGE 
    standard_scaler = StandardScaler()
    # robust_scaler = RobustScaler()        # RobustScaler ignores outliers 
    X = standard_scaler.fit_transform(X)




    # Train-test split
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)

    print(X)
    print(y)

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    log_reg.fit(X_train, y_train.ravel())

    predictions = log_reg.predict(X_test)

    score = log_reg.score(X_test, y_test.ravel())
    print(score)

    cm = metrics.confusion_matrix(y_test.ravel(), predictions)
    print(cm)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(9,9))
    # sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    # plt.ylabel('Actual label');
    # plt.xlabel('Predicted label');
    # all_sample_title = 'Accuracy Score: {0}'.format(score)
    # plt.title(all_sample_title, size = 15);
    # plt.show()

    return X, y


if __name__ == "__main__":
    read_credit_card_file()
