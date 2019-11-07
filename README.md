# Project for Machine learning for Michael Bitney and Magnus Grøndalen
This is a machine learning project where process credit card data and use machine learning to extrapolate patterns and other data.

Here we have built two classes, one called Regression_package and one called NeuralNet_package.
The Regression_package contains Linear Regression which has 3 methods (OLS,Ridge,Lasso) and LogisticRegressor which performs a gradient descent. The Linear Regression methods are taking from a previous project (https://www.google.com "2") #trying to get a link to work.

The NeuralNet_package contains a Logistic NeuralNet and regression NeuralNet. NeuralNet_package is designed to be very similiar to how you would use scikitlearns package on NeuralNet


## Running the scripts
In order to run the python files you can simply type in the terminal:

```
python main_*.py
```

where the * will replace the subtask you want to run

If you want to run all of them and you have access to a terminal that can run
a bash-script, you can do this by running:

```
./run_scripts/run_all.sh
```

```
python additional_files/displaying_data.py

```

We have a test file that runs multiple tests on the RegressionMethod.
You can run this by typing:

```
pytest -v
```
This will test both packages with test on for example R2-score, sigmoidfunction and so on.



### Authors

* **Michael Bitney**
* **Magnus Grøndalen**

## More will come
