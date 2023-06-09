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

print('island list: ', island_encoder.classes_)
x1= input("Enter island >> ")  #['Torgersen']

while ([x1] not in island_encoder.classes_):
    print("Try again, select from this list: ", island_encoder.classes_)
    x1= input('Enter island >> ') 
    
x1= island_encoder.transform([x1])[0]

x2 = float(input('Enter culmen length (mm) >> ')) #37.0
x3 = float(input('Enter culmen depth (mm) >> ')) #19.3
x4 = float(input('Enter flipper length (mm) >> ')) #192.3
x5 = float(input('Enter body mass (g) >> ')) #3750.0

print('sex list: ',sex_encoder.classes_ )
x6 = input('Enter sex >> ')  # ['MALE']
while ([x6] not in sex_encoder.classes_):
    print("Try again, select from this list: ", sex_encoder.classes_)
    x6= input('Enter sex >> ') 
x6 = sex_encoder.transform([x6])[0]

x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6]).reshape(1,-1), 
             columns=['island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex'])

pred = model.predict(x_new)
print('Predicted Species: ' , species_encoder.inverse_transform(pred)[0])
