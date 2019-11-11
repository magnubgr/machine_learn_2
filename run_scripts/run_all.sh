#!/bin/bash
#if [ "$#" == "0" ]; then
#  echo 'please write in the number n in command line, like this:'
#  echo 'plotting_ade_lambda.sh n'
#  exit 1
#fi


echo "1"
python logreg_plots.py
echo "2"
python logreg_comparing_performance_sklearn.py
echo "3"
python neuralnet_clf_performance_test.py
echo "4"
python neuralnet_reg_comparison_sklearn.py
# echo "5"
# python main.py
