# Machine learning project 2 by Michael Bitney and Magnus Grøndalen
This is the repository for a machine learning project at the University of Oslo, where we process credit card data and use machine learning to extrapolate patterns and other data. We create a logistic regressor for classification of the credit card data, and create a neural network from scratch to perform both classification and regression analysis.

Here we have made two packages; one called Regression_package and one called NeuralNet_package.
The Regression_package contains Linear Regression which has 3 methods (OLS, Ridge and Lasso) and LogisticRegressor which uses a gradient descent to minimize the loss function. The Linear Regression methods are taking from a [previous project](https://github.uio.no/michaesb/ml_project1_mms "2").

The NeuralNet_package contains a classifying NeuralNet and regression NeuralNet. NeuralNet_package is designed to be very similiar to how you would use Scikit-Learns package on NeuralNet.


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


We have a test file that runs multiple tests on the Regression_package and NeuralNet_package.
You can run this by typing:
```
pytest -v
```
This will test both packages with test on for example R2-score, sigmoidfunction and so on.


Our rapport is the file called report2_fysstk_magnubgr_michaesb.pdf and here we present our findings



### Authors

* **Michael Bitney**
* **Magnus Grøndalen**
