import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('diabetes_model.pkl', 'rb'))
model3 = pickle.load(open('stroke_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/stroke')
def stroke():
    return render_template('stroke.html')

@app.route('/predictheart',methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "trestbps","chol","thalach", "oldpeak", "sex_0",
                       "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3","  fbs_0",
                        "restecg_0","restecg_1","restecg_2","exang_0","exang_1",
                        "slope_0","slope_1","slope_2","ca_0","ca_1","ca_2","thal_1",
                        "thal_2","thal_3"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** heart disease (Please consult a Doctor) **"
    else:
        res_val = "no heart disease "
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


@app.route('/predictstroke', methods=['POST'])
def predictstroke():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "hypertension", "disease",
                     "glucose", "bmi", "gender_male",
                     "gender_other", "married_yes", "work_type_Never_worked",
                     "work_type_Private", "work_type_Self_employed", "work_type_children",
                     "Residence_type_Urban","smoking_status_formerly_smoked",
                     "smoking_status_never_smoked", "smoking_status_smokes"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "** stroke (Please consult a Doctor) **"
    else:
        res_val = "No Stroke  "

    return render_template('stroke.html', prediction_text2='Patient has {}'.format(res_val))

@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["Pregnancies", "Glucose", "BloodPressure",
                     "SkinThickness", "Insulin", "BMI",
                     "DiabetesPedigreeFunction", "Age"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "** U have a Diabetes **"
    else:
        res_val = "No Diabetes "

    return render_template('diabetes.html', prediction_text3='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
