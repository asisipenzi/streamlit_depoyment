import streamlit as st
import pandas as pd
import pickle
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from pprint import pprint
import os 


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()



if os.path.exists:
    final_model = pickle.load(open('data/final_nn_pickle.pkl', "rb"))
    food_model = final_model["best_model"]

#with open('data/final_nn_pickle.pkl', "rb") as file:
 #   pickle.dump(file)
#print("finishing training new  model...")



print('----------------------------------------')
pprint(food_model)
print('----------------------------------------')

def get_data(filename):
    retrieved_data = pd.read_csv(filename)
    return retrieved_data

def predict_from_model(carbohydrates, protein, vitamin, fat_content, minerals):

    input = np.array(
        [[carbohydrates, protein, vitamin, fat_content, minerals]]).astype(np.float64)
    prediction = food_model.predict(input)

    return prediction


with header:
    st.title('Welcome to Your Personal Nutritionist!')
    st.text('In this project, we hope to show how food serving sizes based on their nutritional value. ')


with dataset:
    st.header('Branded food nutrient datasets')
    st.text('We used the standard dataset from USDA Branded Food Products Database.')

    branded_data = get_data('data/branded_food.csv')
    food_data = get_data('data/food.csv')
    nutrient_data = get_data('data/nutrient.csv')

    cleaned_data = get_data('data/clean_foods_df.csv')

    branded = branded_data.head(15)
    food = food_data.head(20)
    nutrient = nutrient_data.head(20)

    cleaned = cleaned_data.head(20)

    st.subheader('Branded foods dataset')
    st.bar_chart(branded)

    st.subheader('Foods Dataset')
    st.bar_chart(food)

    st.subheader('Nutrients Dataset')
    st.bar_chart(nutrient)

    st.subheader('Cleaned Dataset')
    st.bar_chart(cleaned)

with features:
    st.header('The features in our model')
    st.text("The features are that we suggest the amount of food to be taken based on the calories in each nutrient on the food packaging.")
#	for list use st.markdown
#	making columns for user input as well as display of suggestion
#	sel_col = st.sidebar()

    gender_UI = st.selectbox('What is your gender?', options=[
        'Male', 'Female', 'Other'])
    user_age = st.text_input('What is your current age?')
    max_carb = st.slider('How much carbohydrates is on your current product?',
                         min_value=5, max_value=150, value=20, step=10)
    max_prot = st.slider('How much protein is on your current product?',
                         min_value=5, max_value=150, value=20, step=10)
    max_vit = st.slider('How much vitamin is on your current product?',
                        min_value=5, max_value=150, value=20, step=10)
    max_fat = st.slider('What is the fat content in your current product?',
                        min_value=5, max_value=150, value=20, step=10)
    max_mineral = st.slider('How much mineral nutrient is on your current product?',
                            min_value=5, max_value=150, value=20, step=10)

with modelTraining:
    st.header('Time to test model training')

if st.button('Predict'):
    result = predict_from_model(
        max_carb, max_prot, max_vit, max_fat, max_mineral)

    st.success('Our Result is: ' + result)

st.write('Based on your input, the mean squared error is:')
st.write('Based on your input, the mean squared error is:')
st.write('Based on your input, the mean squared error is:')
