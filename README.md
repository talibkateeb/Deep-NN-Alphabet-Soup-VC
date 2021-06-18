# Deep-NN-Alphabet-Soup-VC

The goal of this project is to build a deep neural net to predict the if the applicant to Alphabet Soup's VC will be a successful startup. 

In this project I accomplished the following: 

* Preprocess data for a neural network model.

* Use the model-fit-predict pattern to compile and evaluate a binary classification model with 2 hidden layers. 

* Optimize the model with three different alternatives. The first by removing a hidden layer, the second by doubling the number of Epochs, and the third by doubling the number of Epochs and adding a hidden layer. 

Accuracy of all the models was around 0.73 and the loss function was around 0.56. Both values barely changed as I optimized the neural net and tried different models.

---

## Technologies

This application runs on python version 3.7, with the following add-ons:

* [Jupyter Lab/Notebook](https://jupyter.org/) - A development tool for interactive programming.

* [Pandas](https://pandas.pydata.org/) - A python librart for data analysis and manipulation.

* [scikit-learn](https://scikit-learn.org/) - A python library for Machine Learning tools.

* [TensorFlow](https://www.tensorflow.org/) - An open source machine learning platform.

* [Keras](https://keras.io/) - An API added onto TensorFlow for deep learning.

---

## Installation Guide

Download and install [Anaconda](https://www.anaconda.com/products/individual-b)

Open terminal and install the required libraries by running the following commands:

    pip install pandas

    pip install -U scikit-learn

    pip install --upgrade tensorflow

Open Anaconda and run Jupyter lab. Go to project directory and open the file named:

    venture_funding_with_deep_learning.ipynb

Click on each cell to run individually:

    Shift + Enter

---

## Example

Running this cell builds the third alternative model for the deep neural net, 100 Epochs and 3 hidden layers:

![Code Example]()

---

## Contributors

*  Talib Kateeb

---

## License

[Click Here To View](https://github.com/talibkateeb/Deep-NN-Alphabet-Soup-VC/blob/main/LICENSE)
