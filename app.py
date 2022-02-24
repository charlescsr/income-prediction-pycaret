from imp import load_module
import gradio as gr
import pandas as pd

from pycaret.classification import *

df = pd.read_csv("adult_trimmmed.csv")

model = load_model('Income Prediction Model')

# List of unique features in workclass column in df
workclass_list = df['workclass'].unique()

# List of unique features in education column in df
education_list = df['education'].unique()

marital_list = {"Single": 0, "Married": 1}

# List of unique features in occupation column in df
occupation_list = df['occupation'].unique()

# List of unique features in relationship column in df
relationship_list = df['relationship'].unique()

sex_list = ["Male", "Female"]

def income_predict(age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week):
    pass