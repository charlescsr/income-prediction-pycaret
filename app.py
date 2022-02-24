from imp import load_module
import gradio as gr
import pandas as pd

from pycaret.classification import *

df = pd.read_csv("adult_trimmmed.csv")

model = load_model('Income Prediction Model')

# List of unique features in workclass column in df
workclass_list = df['workclass'].unique().tolist()

# List of unique features in education column in df
education_list = df['education'].unique().tolist()

marital_list = ["Single", "Married"]
marital_dict = {0: "Single", 1: "Married"}

# List of unique features in occupation column in df
occupation_list = df['occupation'].unique().tolist()

# List of unique features in relationship column in df
relationship_list = df['relationship'].unique().tolist()

sex_list = ["Male", "Female"]

def income_predict(age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week):
    feats = [age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week]

    feats_s = pd.Series(feats)

    print(feats_s)

    return feats_s


gr.Interface(income_predict, inputs=[
    "number", gr.inputs.Dropdown(workclass_list), "number", gr.inputs.Dropdown(education_list), 
    gr.inputs.Dropdown(marital_list), gr.inputs.Dropdown(occupation_list), gr.inputs.Dropdown(relationship_list), 
    gr.inputs.Dropdown(sex_list), 

])