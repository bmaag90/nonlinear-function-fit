# Nonlinear Function Fit 

## Description

A simple app built with [streamlit.io](https://streamlit.io/) that performs nonlinear function fitting using different methods.

Currently supported methods:
1. Locally weighted regression
2. k-Nearest Neighbours
3. Neural network

<img src="https://github.com/bmaag90/nonlinear-function-fit/blob/main/img/screenshot_1.png" width=33% height=33%>

Each method is being trained on a randomly generated dataset (with added noise) and evaluated on equally distanced samples of a specified range of x-values.
The respective results are plotted in a plotly-generated graph.

## How to use

### Run the app
To run the app, open a command line and use the following command

```bash
streamlit run app.py
```

### Features

<img src="https://github.com/bmaag90/nonlinear-function-fit/blob/main/img/screenshot_2.png" width=33% height=33%>

1. Specify a function, which you would like to approximate
2. Select a specific method to fit the function on some randomly generated training data
3. Predicts the function within a specified range of x-values (via sidebar configurations)
4. Show the predictions of the neural network at different epochs during the training process
