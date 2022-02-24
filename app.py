import streamlit as st
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


def income_predict(age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week):
    feats = [age, working_class, final_weight, education, marital_status, occupation, relationship, sex, capital_gain, capital_loss, hours_per_week]

    feats_s = pd.Series(feats)
    feats_s = feats_s.values
    feats_s = feats_s.reshape(1, -1)

    # Convert feats_s into Pandas Data Frame
    feats_df = pd.DataFrame(feats_s, columns=feats_cols)

    pred = predict_model(model, data=feats_df)

    # Return the predicted income and score rounded to 2 decimal places
    return "Income is " + str(pred.loc[0, 'Label']) + " with score of " + str(round(pred.loc[0, 'Score'] * 100, 2)) + "%"


def main():
    st.title("Income Prediction App using PyCaret")
    st.text("This app predicts if the income is more than 50k or less than 50k")
    st.text("")

    age = st.number_input("Age", min_value=0, max_value=100)
    working_class = st.selectbox("Working Class", workclass_list)
    final_weight = st.number_input("Final Weight", min_value=1)
    education = st.selectbox("Education", education_list)
    marital_status = st.selectbox("Marital Status", marital_list)
    occupation = st.selectbox("Occupation", occupation_list)
    relationship = st.selectbox("Relationship", relationship_list)
    sex = st.selectbox("Sex", sex_list)
    gain = st.number_input("Capital Gain", min_value=0)
    loss = st.number_input("Capital Loss", min_value=0)
    hours = st.number_input("Hours Per Week", min_value=1)

    res = ""

    if st.button("Predict"):
        res = income_predict(age, working_class, final_weight, education, marital_status, occupation, relationship, sex, gain, loss, hours)

        st.success(res)

if __name__ == '__main__':
    main()