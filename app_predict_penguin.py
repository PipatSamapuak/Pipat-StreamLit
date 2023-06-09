
import streamlit as st 
from streamlit_option_menu import option_menu
import plotly.graph_objects as px
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


model = pickle.load(open('model.penguins.sav','rb'))
species_encoder = pickle.load(open('encoder.speices.sav','rb'))
island_encoder = pickle.load(open('encoder.island.sav','rb'))
sex_encoder = pickle.load(open('encoder.sex.sav','rb'))
evaluations = pickle.load(open('evals.all.sav','rb'))

st.title('Penguin Species Prediction')

x1 = st.radio('Select island', island_encoder.classes_)
x1 = island_encoder.transform([x1])[0]
x2 = st.slider('Select culmen length(mm)', 25, 70, 40)
x3 = st.slider('Select culmen depth(mm)', 10, 30, 15)
x4 = st.slider('Select flipper length(mm)', 150, 250, 200)
x5 = st.slider('Select body mass(g)', 2500, 6500, 3200) 
x6 = st.radio('Select sex', sex_encoder.classes_)
x6 = sex_encoder.transform([x6])[0]

x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6]).reshape(1,-1), 
                 columns=['island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex'])

pred = model.predict(x_new)

st.write('Predicted Species: ' , species_encoder.inverse_transform(pred)[0]) 
