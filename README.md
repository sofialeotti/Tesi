# Instructions to run the code
## Prerequisites
To run this code the following libraries are needed: 
  - [Pandas](https://pandas.pydata.org/)
  - [Numpy](https://numpy.org/install/)
  - [scikit-learn](https://scikit-learn.org/stable/install.html)
  - [Tensorflow](https://www.tensorflow.org/install)
  - [Spektral](https://graphneural.network/)
# How to run the code
## Analysis ##
Inside the `Analysis` directory

```
python3 analysis.py
```
The output files will be saved in the `plots` directory.
## Graph Neural Network ##
Inside the `NeuralNetwork` directory

```
python3 main.py GNN
```
The evaluation results (confusion matrix, ROC curve, accuracy plot and loss plot) and the feature importance histogram will be saved in the `evaluation_results` directory. The model will be saved in the `saved_models` directory. The evaluation metrics and the training time are displayed in the terminal.
