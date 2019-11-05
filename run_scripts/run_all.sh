#!/bin/bash
#if [ "$#" == "0" ]; then
#  echo 'please write in the number n in command line, like this:'
#  echo 'plotting_ade_lambda.sh n'
#  exit 1
#fi


echo "1"
python main_logistic_reg.py
echo "2"
python main_comparing_logistic.py
echo "3"
python main_neuralnet.py
echo "4"
python main_sklearn_neuralnet.py
# echo "5"
# python main.py
