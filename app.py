from imp import load_module
import gradio as gr
import pandas as pd

from pycaret.classification import *

df = pd.read_csv("adult_trimmmed.csv")

model = load_model('Income Prediction Model')

