import gradio as gr
import pandas as pd
import numpy as np

from pycaret.classification import *

df = pd.read_csv("adult_trimmed.csv")

feats_cols = df.columns.tolist()
feats_cols.remove("income")

model = load_model('Income Prediction Model')

# List of unique features in workclass column in df
workclass_list = df['workclass'].unique().tolist()
workclass_list.remove(np.nan)

# List of unique features in education column in df
education_list = df['education'].unique().tolist()

marital_list = ["Single", "Married"]

# List of unique features in occupation column in df
occupation_list = df['occupation'].unique().tolist()
occupation_list.remove(np.nan)

# List of unique features in relationship column in df
relationship_list = df['relationship'].unique().tolist()

sex_list = ["Male", "Female"]

print(df.iloc[1, :])

def income_predict(age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week):
    feats = [age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week]

    feats_s = pd.Series(feats)
    feats_s = feats_s.values
    feats_s = feats_s.reshape(1, -1)

    # Convert feats_s into Pandas Data Frame
    feats_df = pd.DataFrame(feats_s, columns=feats_cols)

    #print(feats_df.iloc[0, :])

    pred = predict_model(model, data=feats_df)

    return "Income is " + str(pred.loc[0, 'Label']) + " with score of " + str(pred.loc[0, 'Score'] * 100) + "%"


gr.Interface(income_predict, inputs=[
    "number", gr.inputs.Dropdown(workclass_list), "number", gr.inputs.Dropdown(education_list), 
    gr.inputs.Dropdown(marital_list, type="index"), gr.inputs.Dropdown(occupation_list), gr.inputs.Dropdown(relationship_list), 
    gr.inputs.Dropdown(sex_list), "number", "number", "number"], 
    outputs="text").launch()