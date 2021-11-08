import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from utils import *
from nn_keras import NN_Keras
from locally_weighted_regression import *

# Set page layout to wide
#st.set_page_config(layout="wide")
# Title of the page
st.title('Non-Linear Function Fit')

# Create sidebar: Dataset settings
st.sidebar.title('Data set')
x_start = st.sidebar.number_input('X - min', value=0.0)
x_end = st.sidebar.number_input('X - max', value=10.0)
x_step = st.sidebar.number_input('X - step', value=0.1)
m = st.sidebar.number_input('Number of samples', value=200)
noise_mu = st.sidebar.number_input('Gaussian noise - mean', value=0.0)
noise_sigma = st.sidebar.number_input('Gaussian noise - standard deviation', value=0.2)

# Text-input: allows to specify the function we want to approximate
str_fx = st.text_input('Function to approximate (must be correct python syntax using numpy as np)', '0.5*x + np.sin(0.5*x)')
X, y, X_linspace = create_dataset(x_start, x_end, x_step, noise_mu, noise_sigma, m, str_fx)

# Select approach to approximate
option = st.selectbox('Select approach',
                      ('Locally Weighted Regression', 'Neural Network'))

# Initial option - locally weighted regression
if option == 'Locally Weighted Regression':
    st.header('Locally Weighted Regression')

    st.write('For more information, please have a look at: Locally Weighted Learning by Peter Englert')
    # Parameter: tau - the kernel width
    tau =  st.number_input('Tau - kernel width', value=0.2)
    # Run model fit
    yhat = predict_weighted_regression(X, X_linspace, y, tau)
    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, 
                            y=y, 
                            mode='markers', 
                            marker_color='rgba(0,0,0,0.5)',
                            name='Training data'))
    fig.add_trace(go.Scatter(x=X_linspace, 
                            y=yhat.squeeze(), 
                            mode='markers', 
                            marker_color='rgba(255,0,0,1)',
                            name='Predictions'))       
    fig.add_trace(go.Scatter(x=X_linspace, 
                            y=eval_fx(X_linspace, str_fx), 
                            mode='lines', 
                            line_color='rgba(51,255,255,0.7)',
                            name='Underlying function'))
    
    st.plotly_chart(fig)

# Second option - neural network
elif option == 'Neural Network':
    st.header('Neural Network')
    st.write('We will use a neural network with one hidden layer (tanh activation to model non-linearities) to approximate the function.')
    # create an NN_Keras object to help training the network
    nn_helper = NN_Keras()
    # scale data
    X_scaled, X_linspace_scaled = nn_helper.scale_data(X, X_linspace)
    # Network parameters
    col1, col2 = st.columns(2)
    with col1:
        epochs =  st.number_input('Training epochs', value=200)
        num_neurons = st.number_input('Number of neurons', value=50)
    with col2:
        learning_rate = st.number_input('Adam - Initial learning rate', value=0.1)

    # Train the model
    model_trained = False
    if st.button('Train model'):
        st.write('Model is being trained...')
        # Delete any old checkpoints
        nn_helper.delete_existing_model_checkpoints()
        # Initialize the model
        nn_helper.init_model(X_scaled, num_neurons, learning_rate)
        # Fit the model with the data
        nn_helper.fit(X_scaled, y, epochs)
        # predict the final function fit
        yhat = nn_helper.predict(X_linspace_scaled)
        model_trained = True
        st.write('Model finished training!')

    # if model has been successfully trained, show the plot
    if model_trained:
        # Generate model predictions at different checkpoints
        df = nn_helper.predict_at_checkpoints(X_linspace)
        # show plot
        fig = px.scatter(df, 
                 x='X', 
                 y='Predictions', 
                 animation_frame='epoch',
                 labels='Predictions',
                 color_discrete_sequence=['red']) 
        fig['data'][0]['name'] = 'Predictions'
        fig['data'][0]['showlegend'] = True
        fig.add_trace(go.Scatter(x=X, 
                                y=y, 
                                mode='markers', 
                                marker_color='rgba(0,0,0,0.5)',
                                name='Training data'))
        fig.add_trace(go.Scatter(x=X_linspace, 
                                    y=eval_fx(X_linspace, str_fx), 
                                    mode='lines', 
                                    line_color='rgba(51,255,255,0.7)',
                                    name='Underlying function'))  
        st.plotly_chart(fig)
        st.write('Press the play button to animate the model predictions at different epochs of the training process.')