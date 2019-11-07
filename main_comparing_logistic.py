#!/usr/bin/env python
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from Regression_package.LogisticRegressor import LogisticRegression
import sklearn.linear_model as skl


"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
clf = LogisticRegression()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y, test_size=0.3, random_state=4)

def heat_map():
    # learning_rate = np.arange(0,1,0.5)
    print("creating heatmap of the accuracy vs learning_rate\
    and # of iterations ")
    learning_rate = np.linspace(0.01,2.5,5)
    n_iterations = np.linspace(100,500,5, dtype=int)

    accuracy_score = np.zeros((len(learning_rate),len(n_iterations)))

    for i in range(len(learning_rate)):
        if 100*i%len(learning_rate) == 0:
            print(int(100*i/len(learning_rate)), "%")
        for j in range(len(n_iterations)):
            clf.fit_data(X_train, 
                        y_train,
                        learning_rate=learning_rate[i], 
                        n_iter=n_iterations[j])
            pred = clf.predict(X_test)
            accuracy = clf.accuracy(pred, y_test.flatten())
            accuracy_score[i,j]=accuracy
    heat_map = sb.heatmap(accuracy_score, xticklabels=n_iterations, yticklabels=learning_rate,  cmap="viridis")
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    plt.title(r"Heatmap of accuracy_score for different learning_rate and iterations", size=20)
    plt.xlabel(r"learning_rate $\gamma $ ", size=18)
    plt.ylabel(r"n_iterations ", size=18)
    plt.show()
    print( np.max(accuracy_score) ,np.where(np.max(accuracy_score)==accuracy_score)[0:1])
    print(np.shape(accuracy_score))
    # print(5,6,learning_rate[5], n_iterations[6])



def learning_rate_plot():
    print ("plotting accuacry as a function of learning_rate")
    learning_rate = np.linspace(0.001,2,50)
    n_iterations = 200
    accuracy_score = np.zeros(len(learning_rate))
    for i in range(len(learning_rate)):
        print(100*i/len(learning_rate),'%')
        clf.fit_data(X_train, 
                    y_train,
                    learning_rate=learning_rate[i], 
                    n_iter=n_iterations
                    )
        pred = clf.predict(X_test)
        accuracy_score[i]= clf.accuracy(pred, y_test.flatten())

    plt.plot(learning_rate, accuracy_score, "*")
    plt.plot(learning_rate, accuracy_score)
    plt.title(r"The accuracy for different learning rate at n_iterations ="+str(n_iterations), size=20)
    plt.xlabel(r"learning_rate $\gamma $", size=18)
    plt.ylabel(r"accuracy ", size=18)

def printing_accuracy(learning_rate, n_iterations):
    clf.fit_data(X_train, y_train,
    learning_rate=learning_rate, n_iter=n_iterations)
    pred = clf.predict(X_test)
    accuracy = clf.accuracy(pred, y_test.flatten())
    print(f"accuracy_score: {accuracy} with learning rate"+ \
        "at {learning_rate} and # iterations {n_iterations}")

def n_iterations_plot():
    print ("plotting accuracy as a function of the number of iterations")
    n_iterations = np.arange(2,200,5)
    learning_rate = 1.5
    print("running accuracy vs iterations plot. learning_rate = "+str(learning_rate))
    accuracy_score = np.zeros(len(n_iterations))

    for j in range(len(n_iterations)):
        print(100*j/len(n_iterations),'%')
        clf.fit_data(X_train, y_train,
        learning_rate=learning_rate, n_iter=n_iterations[j])
        pred = clf.predict(X_test)
        accuracy = clf.accuracy(pred, y_test.flatten())
        accuracy_score[j]=accuracy



    plt.plot(n_iterations, accuracy_score, label="Classifier package")
    plt.title(r"The Classifier package vs ScikitLearn", size=20)
    plt.xlabel(r"n_iterations ", size=18)
    plt.ylabel(r"accuracy ", size=18)

def scikit_LogReg_n_iter():
    ##### Scikit-Learn Logistic regression #####
    print("running accuracy vs iterations plot. ScikitLearn")
    n_iterations = np.arange(2,200,5)
    accuracy_score = np.zeros(len(n_iterations))
    for i in range(len(n_iterations)):
        if 100*i%len(n_iterations) == 0:
            print(int(100*i/len(n_iterations)), "%")
        log_reg = skl.LogisticRegression(random_state=0,
                                     solver='lbfgs',
                                     multi_class='multinomial',
                                     max_iter=n_iterations[i])
        log_reg.fit(X_train, y_train.flatten())

        prediction = log_reg.predict(X_test)

        accuracy_score[i] = clf.accuracy(prediction,y_test)


    plt.plot(n_iterations, accuracy_score, label="scikitlearn")
    plt.title(r"The accuracy for different n_iterations for Scikitlearn", size=20)
    plt.xlabel(r"n_iterations ", size=18)
    plt.ylabel(r"accuracy ", size=18)

# n_iterations_plot()
# scikit_LogReg_n_iter()
# learning_rate_plot()
heat_map()
# plt.legend()
# plt.show()
