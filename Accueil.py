                                                            # IMPORTATION DES LIBRAIRIES

import streamlit as st
from PIL import Image
import numpy as np
import requests
import joblib
from io import BytesIO
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from catboost import CatBoostClassifier
import lightgbm as lgb
warnings.filterwarnings("ignore")

                                                            # MISE EN PAGE

# Configuration de la page
st.set_page_config(page_title="Clinical Data Insight", page_icon=":dna:", layout="wide")

#st.markdown("""
#<style>
#    header, footer {visibility: hidden;}
#</style>
#""",unsafe_allow_html=True)

# Mise en place du background de l'appli
link_background = "https://img.freepik.com/free-vector/abstract-hexagonal-shapes-banner-blue-color_1017-25909.jpg?w=1380&t=st=1707294655~exp=1707295255~hmac=060384988f7680caec7793d42e99d94b80252748acc877912eb0aed5f4088f34"
def set_bg_hack_url():

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url({link_background});
             background-size:cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

link_logo = "https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/Logo_Clinical_Data_Insight_V2.png"
st.sidebar.image(Image.open("logos/Logo_Clinical_Data_Insight_V2.png"), width=280, use_column_width=False)

# Masquage des info-bulles
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''
st.markdown(hide_img_fs, unsafe_allow_html=True)

# Boutons dans la sidebar à gauche
selected_page = st.sidebar.radio("Navigation", ["Accueil", "Informations", "Cancer du sein", "Diabète", "Maladies cardiaques", "Maladies du foie", "Maladie rénale chronique"], label_visibility="collapsed")
st.sidebar.write(" ")
text_disclaimer = "Les prédictions générées par l'application ne peuvent en aucun cas se substituer à l'avis d'un professionnel de la santé.<br><br>Leur but est de fournir un support complémentaire lors du processus de prise de décision concernant d'éventuels examens complémentaires et traitements.<br><br>Il est crucial de souligner que les données que vous saisissez sont entièrement anonymisées et ne font l'objet d'aucune conservation, en stricte conformité avec les directives rigoureuses du Règlement Général de Protection des Données (RGPD)."
st.sidebar.markdown(f'''**[DISCLAIMER]**<br>{text_disclaimer}''', unsafe_allow_html=True)

# Définir une fonction pour le contenu de chaque page
def maladies_cardiaques():

    st.write('## Maladies cardiaques')
    st.write('#### Renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Chargement du modèle sauvegardé
    model_heart_disease_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/Gaussian_HeartDisease.joblib'
    response = requests.get(model_heart_disease_url)
    response.raise_for_status()
    model_heart_disease = joblib.load(BytesIO(response.content))
    
    scaler_heart_disease_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/ScalePwrTransf_HeartDisease.joblib'
    response = requests.get(scaler_heart_disease_url)
    response.raise_for_status()
    scaler_heart_disease = joblib.load(BytesIO(response.content))

    if "reset" not in st.session_state:
        st.session_state.reset = False

    default_values = {"age": 0, "sex": 0, "cp": 0.0000, "trestbps": 0.0000, "chol":0.0000,"fbs":0.0000,"restecg":0.0000,"thalach":0.0000, "exang":0.0000, "oldpeak":0.0000, "slope":0.0000, "ca":0.0000, "thal":0.0000}

    age = default_values["age"] if st.session_state.reset else st.session_state.get("age", default_values["age"])
    sex = default_values["sex"] if st.session_state.reset else st.session_state.get("sex", default_values["sex"])
    cp = default_values["cp"] if st.session_state.reset else st.session_state.get("cp", default_values["cp"])
    trestbps = default_values["trestbps"] if st.session_state.reset else st.session_state.get("trestbps", default_values["trestbps"])
    chol = default_values["chol"] if st.session_state.reset else st.session_state.get("chol", default_values["chol"])
    fbs = default_values["fbs"] if st.session_state.reset else st.session_state.get("fbs", default_values["fbs"])
    restecg = default_values["restecg"] if st.session_state.reset else st.session_state.get("restecg", default_values["restecg"])
    thalach = default_values["thalach"] if st.session_state.reset else st.session_state.get("thalach", default_values["thalach"])
    exang = default_values["exang"] if st.session_state.reset else st.session_state.get("exang", default_values["exang"])
    oldpeak = default_values["oldpeak"] if st.session_state.reset else st.session_state.get("oldpeak", default_values["oldpeak"])
    slope = default_values["slope"] if st.session_state.reset else st.session_state.get("slope", default_values["slope"])
    ca = default_values["ca"] if st.session_state.reset else st.session_state.get("ca", default_values["ca"])
    thal = default_values["thal"] if st.session_state.reset else st.session_state.get("thal", default_values["thal"])

    col1, col2, col3, col4 = st.columns([2,2,1,2])

    with col1:
        st.session_state.age = st.number_input("Age", value=int(age), step=1, format="%d")
        sex_options = [0, 1]
        st.session_state.sex = st.radio("Genre (0=Femme, 1=Homme)", sex_options, index=sex_options.index(sex), horizontal=1)
        st.session_state.cp = st.number_input("Type de douleur thoracique", value=cp, step=0.0001, format="%.4f")
        st.session_state.trestbps = st.number_input("Pression artérielle au repos", value=trestbps, step=0.0001, format="%.4f")
        st.session_state.chol = st.number_input("Cholestérol sérique en mg/dl", value=chol, step=0.0001, format="%.4f")
        st.session_state.fbs = st.number_input("Taux de sucre dans le sang à jeun > 120 mg/dl", value=fbs, step=0.0001, format="%.4f")
        st.session_state.restecg = st.number_input("Résultats électrocardiographiques au repos", value=restecg, step=0.0001, format="%.4f")
    
    with col2:
        st.session_state.thalach = st.number_input("Fréquence cardiaque maximale atteinte", value=thalach, step=0.0001, format="%.4f")
        st.session_state.exang = st.number_input("Angine induite par l'exercice", value=exang, step=0.0001, format="%.4f")
        st.session_state.oldpeak = st.number_input("Dépression de ST induite par l'exercice par rapport au repos", value=oldpeak, step=0.0001, format="%.4f")
        st.session_state.slope = st.number_input("Pente du segment ST à l'exercice", value=slope, step=0.0001, format="%.4f")
        st.session_state.ca = st.number_input("Nombre de vaisseaux principaux colorés par la fluoroscopie", value=ca, step=0.0001, format="%.4f")
        st.session_state.thal = st.number_input("Résultat thallium scintigraphique", value=thal, step=0.0001, format="%.4f")

    with col3:
        st.write(" ")

    with col4:
        st.write(" ")
    # Reset button
        if st.button("Réinitialiser les valeurs"):
            st.session_state.reset = True
            st.session_state.age = default_values["age"]
            st.session_state.sex = default_values["sex"]
            st.session_state.cp = default_values["cp"]
            st.session_state.trestbps = default_values["trestbps"]
            st.session_state.chol = default_values["chol"]
            st.session_state.fbs = default_values["fbs"]
            st.session_state.restecg = default_values["restecg"]
            st.session_state.thalach = default_values["thalach"]
            st.session_state.exang = default_values["exang"]
            st.session_state.oldpeak = default_values["oldpeak"]
            st.session_state.slope = default_values["slope"]
            st.session_state.ca = default_values["ca"]
            st.session_state.thal = default_values["thal"]
        else:
            st.session_state.reset = False

        # Bouton pour lancer le traitement des données
        if st.button("Diagnostic"):
            if all(value == 0 for value in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
                st.warning("## Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
            else:
                my_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
                my_data_scaled = scaler_heart_disease.transform(my_data)
                predictions = model_heart_disease.predict(my_data_scaled)
                if predictions[0] == 0:
                    st.markdown(f"""**:green[Le modèle prédit que le patient n'a pas de maladie cardiaque.]**""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""**:red[Le modèle prédit que le patient a un risque de développer une maladie cardiaque.]**""", unsafe_allow_html=True)

def maladies_du_foie():

    st.write('## Maladies du foie')
    st.write('#### Renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Chargement du modèle sauvegardé
    model_liver_disease_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/RandomForest_liver.sav'
    response = requests.get(model_liver_disease_url)
    response.raise_for_status()
    model_liver_disease = joblib.load(BytesIO(response.content))
    
    scaler_liver_disease_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/MaxAbsScaler_Liver.joblib'
    response = requests.get(scaler_liver_disease_url)
    response.raise_for_status()
    scaler_liver_disease = joblib.load(BytesIO(response.content))

    if "reset" not in st.session_state:
        st.session_state.reset = False

    default_values = {"age": 0, "gender": 0, "total_bilirubin": 0.0000, "alkaline_phosphotase": 0.0000, "alamine_aminotransferase":0.0000,"albumin_and_globulin_ratio":0.0000}

    age = default_values["age"] if st.session_state.reset else st.session_state.get("age", default_values["age"])
    gender = default_values["gender"] if st.session_state.reset else st.session_state.get("gender", default_values["gender"])
    total_bilirubin = default_values["total_bilirubin"] if st.session_state.reset else st.session_state.get("total_bilirubin", default_values["total_bilirubin"])
    alkaline_phosphotase = default_values["alkaline_phosphotase"] if st.session_state.reset else st.session_state.get("alkaline_phosphotase", default_values["alkaline_phosphotase"])
    alamine_aminotransferase = default_values["alamine_aminotransferase"] if st.session_state.reset else st.session_state.get("alamine_aminotransferase", default_values["alamine_aminotransferase"])
    albumin_and_globulin_ratio = default_values["albumin_and_globulin_ratio"] if st.session_state.reset else st.session_state.get("albumin_and_globulin_ratio", default_values["albumin_and_globulin_ratio"])


    col1, col2, col3, col4 = st.columns([2,2,1,2])

    with col1:
        st.session_state.age = st.number_input("Age", value=int(age), step=1, format="%d")
        st.session_state.alkaline_phosphotase = st.number_input("Phosphatase alcaline", value=alkaline_phosphotase, step=0.0001, format="%.4f")
        st.session_state.total_bilirubin = st.number_input("Bilirubine totale", value=total_bilirubin, step=0.0001, format="%.4f")

    with col2:
        gender_options = [0, 1]
        st.session_state.gender = st.radio("Genre (0=Femme, 1=Homme)", gender_options, index=gender_options.index(gender), horizontal=0)
        st.session_state.alamine_aminotransferase = st.number_input("Alamine aminotransférase", value=alamine_aminotransferase, step=0.0001, format="%.4f")
        st.session_state.albumin_and_globulin_ratio = st.number_input("Rapport albumine et globuline", value=albumin_and_globulin_ratio, step=0.0001, format="%.4f")

    with col3:
        st.write(" ")
    
    with col4:
        st.write(" ")
    # Reset button
        if st.button("Réinitialiser les valeurs"):
            st.session_state.reset = True
            st.session_state.age = default_values["age"]
            st.session_state.gender = default_values["gender"]
            st.session_state.total_bilirubin = default_values["total_bilirubin"]
            st.session_state.alkaline_phosphotase = default_values["alkaline_phosphotase"]
            st.session_state.alamine_aminotransferase = default_values["alamine_aminotransferase"]
            st.session_state.albumin_and_globulin_ratio = default_values["albumin_and_globulin_ratio"]
        else:
            st.session_state.reset = False

        # Bouton pour lancer le traitement des données
        if st.button("Diagnostic"):
            if all(value == 0 for value in [age, gender, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]):
                st.warning("## Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
            else:
                my_data = np.array([[age, gender, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]])
                my_data_scaled = scaler_liver_disease.transform(my_data)
                predictions = model_liver_disease.predict(my_data_scaled)
                if predictions[0] == 0:
                    st.markdown(f"""**:green[Le modèle prédit que le patient n'a pas de maladie du foie.]**""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""**:red[Le modèle prédit que le patient a un risque de développer une maladie du foie.]**""", unsafe_allow_html=True)

def maladie_renale_chronique():

    st.write('## Maladies rénale chronique')
    st.write('#### Renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Chargement du modèle sauvegardé
    model_ckd_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/RandomForest_CKD.sav'
    response = requests.get(model_ckd_url)
    response.raise_for_status()
    model_ckd = joblib.load(BytesIO(response.content))
    
    scaler_ckd_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/StandardScaler_CKD.joblib'
    response = requests.get(scaler_ckd_url)
    response.raise_for_status()
    scaler_ckd = joblib.load(BytesIO(response.content))

    if "reset" not in st.session_state:
        st.session_state.reset = False

    default_values = {"Age": 0, "Specific_Gravity": 0.0000, "Albumin": 0.0000, "Sugar": 0.0000, "Red_Blood_Cells": 0.0000, "Pus_Cell":0.0000,"Pus_Cell_Clumps":0.0000,
                      "Bacteria":0.0000,"Blood_Glucose_Random":0.0000, "Blood_Urea":0.0000, "Serum_Creatinine":0.0000, "Sodium":0.0000, "Potassium":0.0000, 
                      "Haemoglobin":0.0000,"White_Blood_Cell_Count":0.0000, "Red_Blood_Cell_Count":0.0000, "Hypertension":0.0000 , "Diabetes_Mellitus":0.0000 , 
                      "Coronary_Artery_Disease":0.0000 , "Appetite":0.0000 , "Pedal_Edema":0.0000 , "Anemia":0.0000}

    Age = default_values["Age"] if st.session_state.reset else st.session_state.get("Age", default_values["Age"])
    Specific_Gravity = default_values["Specific_Gravity"] if st.session_state.reset else st.session_state.get("Specific_Gravity", default_values["Specific_Gravity"])
    Albumin = default_values["Albumin"] if st.session_state.reset else st.session_state.get("Albumin", default_values["Albumin"])
    Sugar = default_values["Sugar"] if st.session_state.reset else st.session_state.get("Sugar", default_values["Sugar"])
    Red_Blood_Cells = default_values["Red_Blood_Cells"] if st.session_state.reset else st.session_state.get("Red_Blood_Cells", default_values["Red_Blood_Cells"])
    Pus_Cell = default_values["Pus_Cell"] if st.session_state.reset else st.session_state.get("Pus_Cell", default_values["Pus_Cell"])
    Pus_Cell_Clumps = default_values["Pus_Cell_Clumps"] if st.session_state.reset else st.session_state.get("Pus_Cell_Clumps", default_values["Pus_Cell_Clumps"])
    Bacteria = default_values["Bacteria"] if st.session_state.reset else st.session_state.get("Bacteria", default_values["Bacteria"])
    Blood_Glucose_Random = default_values["Blood_Glucose_Random"] if st.session_state.reset else st.session_state.get("Blood_Glucose_Random", default_values["Blood_Glucose_Random"])
    Blood_Urea = default_values["Blood_Urea"] if st.session_state.reset else st.session_state.get("Blood_Urea", default_values["Blood_Urea"])
    Serum_Creatinine = default_values["Serum_Creatinine"] if st.session_state.reset else st.session_state.get("Serum_Creatinine", default_values["Serum_Creatinine"])
    Sodium = default_values["Sodium"] if st.session_state.reset else st.session_state.get("Sodium", default_values["Sodium"])
    Potassium = default_values["Potassium"] if st.session_state.reset else st.session_state.get("Potassium", default_values["Potassium"])
    Haemoglobin = default_values["Haemoglobin"] if st.session_state.reset else st.session_state.get("Haemoglobin", default_values["Haemoglobin"])
    White_Blood_Cell_Count = default_values["White_Blood_Cell_Count"] if st.session_state.reset else st.session_state.get("White_Blood_Cell_Count", default_values["White_Blood_Cell_Count"])
    Red_Blood_Cell_Count = default_values["Red_Blood_Cell_Count"] if st.session_state.reset else st.session_state.get("Red_Blood_Cell_Count", default_values["Red_Blood_Cell_Count"])
    Hypertension = default_values["Hypertension"] if st.session_state.reset else st.session_state.get("Hypertension", default_values["Hypertension"])
    Diabetes_Mellitus = default_values["Diabetes_Mellitus"] if st.session_state.reset else st.session_state.get("Diabetes_Mellitus", default_values["Diabetes_Mellitus"])
    Coronary_Artery_Disease = default_values["Coronary_Artery_Disease"] if st.session_state.reset else st.session_state.get("Coronary_Artery_Disease", default_values["Coronary_Artery_Disease"])
    Appetite = default_values["Appetite"] if st.session_state.reset else st.session_state.get("Appetite", default_values["Appetite"])
    Pedal_Edema = default_values["Pedal_Edema"] if st.session_state.reset else st.session_state.get("Pedal_Edema", default_values["Pedal_Edema"])
    Anemia = default_values["Anemia"] if st.session_state.reset else st.session_state.get("Anemia", default_values["Anemia"])

    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        st.session_state.Age = st.number_input("Age", value=int(Age), step=1, format="%d")
        st.session_state.Specific_Gravity = st.number_input("Gravité spécifique", value=Specific_Gravity, step=0.0001, format="%.4f")
        st.session_state.Albumin = st.number_input("Albumine", value=Albumin, step=0.0001, format="%.4f")
        st.session_state.Sugar = st.number_input("Sucre", value=Sugar, step=0.0001, format="%.4f")
        st.session_state.Red_Blood_Cells = st.number_input("Globules rouges", value=Red_Blood_Cells, step=0.0001, format="%.4f")
        st.session_state.Pus_Cell = st.number_input("Cellules de pus", value=Pus_Cell, step=0.0001, format="%.4f")
    with col2:
        st.session_state.Pus_Cell_Clumps = st.number_input("Amas de cellules de pus", value=Pus_Cell_Clumps, step=0.0001, format="%.4f")
        st.session_state.Bacteria = st.number_input("Bactéries", value=Bacteria, step=0.0001, format="%.4f")
        st.session_state.Blood_Glucose_Random = st.number_input("Glycémie aléatoire", value=Blood_Glucose_Random, step=0.0001, format="%.4f")
        st.session_state.Blood_Urea = st.number_input("Urée sanguine", value=Blood_Urea, step=0.0001, format="%.4f")
        st.session_state.Serum_Creatinine = st.number_input("Créatinine sérique", value=Serum_Creatinine, step=0.0001, format="%.4f")
        st.session_state.Sodium = st.number_input("Sodium", value=Sodium, step=0.0001, format="%.4f")
    with col3:
        st.session_state.Potassium = st.number_input("Potassium", value=Potassium, step=0.0001, format="%.4f")
        st.session_state.Haemoglobin = st.number_input("Hémoglobine", value=Haemoglobin, step=0.0001, format="%.4f")
        st.session_state.White_Blood_Cell_Count = st.number_input("Numération des globules blancs", value=White_Blood_Cell_Count, step=0.0001, format="%.4f")
        st.session_state.Red_Blood_Cell_Count = st.number_input("Numération des globules rouges", value=Red_Blood_Cell_Count, step=0.0001, format="%.4f")
        st.session_state.Hypertension = st.number_input("Hypertension", value=Hypertension, step=0.0001, format="%.4f")
        st.session_state.Diabetes_Mellitus = st.number_input("Diabète sucré", value=Diabetes_Mellitus, step=0.0001, format="%.4f")
    with col4:
        st.session_state.Coronary_Artery_Disease = st.number_input("Maladie coronarienne", value=Coronary_Artery_Disease, step=0.0001, format="%.4f")
        st.session_state.Appetite = st.number_input("Appétit", value=Appetite, step=0.0001, format="%.4f")
        st.session_state.Pedal_Edema = st.number_input("Œdème des membres inférieurs", value=Pedal_Edema, step=0.0001, format="%.4f")
        st.session_state.Anemia = st.number_input("Anémie", value=Anemia, step=0.0001, format="%.4f")

    # Reset button

    if st.button("Réinitialiser les valeurs"):
        st.session_state.reset = True

        st.session_state.Age = default_values["Age"]
        st.session_state.Specific_Gravity = default_values["Specific_Gravity"]
        st.session_state.Albumin = default_values["Albumin"]
        st.session_state.Sugar = default_values["Sugar"]
        st.session_state.Red_Blood_Cells = default_values["Red_Blood_Cells"]
        st.session_state.Pus_Cell = default_values["Pus_Cell"]
        st.session_state.Pus_Cell_Clumps = default_values["Pus_Cell_Clumps"]
        st.session_state.Bacteria = default_values["Bacteria"]
        st.session_state.Blood_Glucose_Random = default_values["Blood_Glucose_Random"]
        st.session_state.Blood_Urea = default_values["Blood_Urea"]
        st.session_state.Serum_Creatinine = default_values["Serum_Creatinine"]
        st.session_state.Sodium = default_values["Sodium"]
        st.session_state.Potassium = default_values["Potassium"]
        st.session_state.Haemoglobin = default_values["Haemoglobin"]
        st.session_state.White_Blood_Cell_Count = default_values["White_Blood_Cell_Count"]
        st.session_state.Red_Blood_Cell_Count = default_values["Red_Blood_Cell_Count"]
        st.session_state.Hypertension = default_values["Hypertension"]
        st.session_state.Diabetes_Mellitus = default_values["Diabetes_Mellitus"]
        st.session_state.Coronary_Artery_Disease = default_values["Coronary_Artery_Disease"]
        st.session_state.Appetite = default_values["Appetite"]
        st.session_state.Pedal_Edema = default_values["Pedal_Edema"]
        st.session_state.Anemia = default_values["Anemia"]

    else:
        st.session_state.reset = False 

    # Bouton pour lancer le traitement des données
    if st.button("Diagnostic"):
        if all(value == 0 for value in [Age, Specific_Gravity, Albumin, Sugar, Red_Blood_Cells, Pus_Cell, Pus_Cell_Clumps, Bacteria, Blood_Glucose_Random, Blood_Urea, Serum_Creatinine, Sodium, Potassium, Haemoglobin, White_Blood_Cell_Count, Red_Blood_Cell_Count, Hypertension, Diabetes_Mellitus, Coronary_Artery_Disease, Appetite, Pedal_Edema, Anemia]):
            st.warning("## Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            my_data = np.array([[Age, Specific_Gravity, Albumin, Sugar, Red_Blood_Cells, Pus_Cell, Pus_Cell_Clumps, Bacteria, Blood_Glucose_Random, Blood_Urea, Serum_Creatinine, Sodium, Potassium, Haemoglobin, White_Blood_Cell_Count, Red_Blood_Cell_Count, Hypertension, Diabetes_Mellitus, Coronary_Artery_Disease, Appetite, Pedal_Edema, Anemia]])
            my_data_scaled = scaler_ckd.transform(my_data)
            predictions = model_ckd.predict(my_data_scaled)
            if predictions[0] == 0:
                st.markdown(f"""**:green[Le modèle prédit que le patient n'a pas de maladie rénale chronique.]**""", unsafe_allow_html=True)
            else:
                st.markdown(f"""**:red[Le modèle prédit que le patient a un risque de développer une maladie rénale chronique.]**""", unsafe_allow_html=True)

def diabete():
    st.write('## Diabète')
    st.write('#### Renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Chargement du modèle sauvegardé
    model_diabete_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/RandomForest_Diabetes.joblib'
    response = requests.get(model_diabete_url)
    response.raise_for_status()
    model_diabete = joblib.load(BytesIO(response.content))

    if "reset" not in st.session_state:
        st.session_state.reset = False

    default_values = {"Pregnancies": 0, "Glucose": 0.0000, "BloodPressure": 0.0000, "SkinThickness": 0.0000, "BMI":0.0000,"Age":0,"DiabetesPedigreeFunction":0.0000,"mean_fractal_dimension":0.0000}

    Pregnancies = default_values["Pregnancies"] if st.session_state.reset else st.session_state.get("Pregnancies", default_values["Pregnancies"])
    Glucose = default_values["Glucose"] if st.session_state.reset else st.session_state.get("Glucose", default_values["Glucose"])
    BloodPressure = default_values["BloodPressure"] if st.session_state.reset else st.session_state.get("BloodPressure", default_values["BloodPressure"])
    SkinThickness = default_values["SkinThickness"] if st.session_state.reset else st.session_state.get("SkinThickness", default_values["SkinThickness"])
    BMI = default_values["BMI"] if st.session_state.reset else st.session_state.get("BMI", default_values["BMI"])
    Age = default_values["Age"] if st.session_state.reset else st.session_state.get("Age", default_values["Age"])
    DiabetesPedigreeFunction = default_values["DiabetesPedigreeFunction"] if st.session_state.reset else st.session_state.get("DiabetesPedigreeFunction", default_values["DiabetesPedigreeFunction"])
    
    col1, col2, col3, col4 = st.columns([2,2,1,2])

    with col1:
        st.session_state.Age = st.number_input("Age", value=int(Age), step=1, format="%d")
        st.session_state.Pregnancies = st.number_input("Nombre de grossesses", value=int(Pregnancies), step=1, format="%d")
        st.session_state.Glucose = st.number_input("Niveau de glucose", value=Glucose, step=0.0001, format="%.4f")
        st.session_state.BloodPressure = st.number_input("Pression artérielle", value=BloodPressure, step=0.0001, format="%.4f")

    with col2:
        st.session_state.SkinThickness = st.number_input("Épaisseur de la peau", value=SkinThickness, step=0.0001, format="%.4f")
        st.session_state.BMI = st.number_input("Indice de masse corporelle", value=BMI, step=0.0001, format="%.4f")
        st.session_state.DiabetesPedigreeFunction = st.number_input("Fonction de pédigrée diabétique", value=DiabetesPedigreeFunction, step=0.0001, format="%.4f")

    with col3:
        st.write(" ")

    with col4:
        # Reset button
        st.write(" ")
        if st.button("Réinitialiser les valeurs"):
            st.session_state.reset = True 
            st.session_state.Pregnancies = default_values["Pregnancies"]
            st.session_state.Glucose = default_values["Glucose"]
            st.session_state.BloodPressure = default_values["BloodPressure"]
            st.session_state.SkinThickness = default_values["SkinThickness"]
            st.session_state.BMI = default_values["BMI"]
            st.session_state.Age = default_values["Age"]
            st.session_state.DiabetesPedigreeFunction = default_values["DiabetesPedigreeFunction"]
        else:
            st.session_state.reset = False

        # Bouton pour lancer le traitement des données
        if st.button("Diagnostic"):
            if all(value == 0 for value in [Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Age, DiabetesPedigreeFunction]):
                st.warning("## Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
            else:
                my_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Age, DiabetesPedigreeFunction]])
                predictions = model_diabete.predict(my_data)
                if predictions[0] == 0:
                    st.markdown(f"""**:green[Le modèle prédit que le patient n'a pas de diabète.]**""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""**:red[Le modèle prédit que le patient a un risque de diabète.]**""", unsafe_allow_html=True)

def cancer_du_sein():
    st.write('## Cancer du sein')
    st.write('#### Renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Chargement du modèle sauvegardé
    model_cancer_breast_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/CastBoost_Cancerbreast.joblib'
    response = requests.get(model_cancer_breast_url)
    response.raise_for_status()
    model_cancer_breast = joblib.load(BytesIO(response.content))
    scaler_cancer_breast_url = 'https://github.com/Ju-stASimpleName/Clinical_Data_Insight/raw/main/joblib/ScalePwrTransf_Cancerbreast.joblib'
    response = requests.get(scaler_cancer_breast_url)
    response.raise_for_status()
    scaler_cancer_breast = joblib.load(BytesIO(response.content))

    if "reset" not in st.session_state:
        st.session_state.reset = False

    default_values = {"mean_radius": 0.0000, "mean_texture": 0.0000, "mean_smoothness": 0.0000, "mean_compactness": 0.0000, "mean_concavity":0.0000,"mean_concave_points":0.0000,"mean_symmetry":0.0000,"mean_fractal_dimension":0.0000}

    mean_radius = default_values["mean_radius"] if st.session_state.reset else st.session_state.get("mean_radius", default_values["mean_radius"])
    mean_texture = default_values["mean_texture"] if st.session_state.reset else st.session_state.get("mean_texture", default_values["mean_texture"])
    mean_smoothness = default_values["mean_smoothness"] if st.session_state.reset else st.session_state.get("mean_smoothness", default_values["mean_smoothness"])
    mean_compactness = default_values["mean_compactness"] if st.session_state.reset else st.session_state.get("mean_compactness", default_values["mean_compactness"])
    mean_concavity = default_values["mean_concavity"] if st.session_state.reset else st.session_state.get("mean_concavity", default_values["mean_concavity"])
    mean_concave_points = default_values["mean_concave_points"] if st.session_state.reset else st.session_state.get("mean_concave_points", default_values["mean_concave_points"])
    mean_symmetry = default_values["mean_symmetry"] if st.session_state.reset else st.session_state.get("mean_symmetry", default_values["mean_symmetry"])
    mean_fractal_dimension = default_values["mean_fractal_dimension"] if st.session_state.reset else st.session_state.get("mean_fractal_dimension", default_values["mean_fractal_dimension"])

    col1, col2, col3, col4 = st.columns([2,2,1,2])

    with col1:
        st.session_state.mean_radius = st.number_input("Rayon moyen de la cellule", value=mean_radius, step=0.0001, format="%.4f")
        st.session_state.mean_texture = st.number_input("Texture moyenne de la cellule", value=mean_texture, step=0.0001, format="%.4f")
        st.session_state.mean_smoothness = st.number_input("Régularité moyenne de la cellule", value=mean_smoothness, step=0.0001, format="%.4f")
        st.session_state.mean_compactness = st.number_input("Compacité moyenne de la cellule", value=mean_compactness, step=0.0001, format="%.4f")
    
    with col2:
        st.session_state.mean_concavity = st.number_input("Concavité moyenne de la cellule", value=mean_concavity, step=0.0001, format="%.4f")
        st.session_state.mean_concave_points = st.number_input("Point concave moyen de la cellule", value=mean_concave_points, step=0.0001, format="%.4f")
        st.session_state.mean_symmetry = st.number_input("Symétrie moyenne de la cellule", value=mean_symmetry, step=0.0001, format="%.4f")
        st.session_state.mean_fractal_dimension = st.number_input("Dimension fractale moyenne de la cellule", value=mean_fractal_dimension, step=0.0001, format="%.4f")

    with col3:
        st.write(" ")

    with col4:
        st.write(" ")
        # Reset button
        if st.button("Réinitialiser les valeurs"):
            st.session_state.reset = True
            st.session_state.mean_radius = default_values["mean_radius"]
            st.session_state.mean_texture = default_values["mean_texture"]
            st.session_state.mean_smoothness = default_values["mean_smoothness"]
            st.session_state.mean_compactness = default_values["mean_compactness"]
            st.session_state.mean_concavity = default_values["mean_concavity"]
            st.session_state.mean_concave_points = default_values["mean_concave_points"]
            st.session_state.mean_symmetry = default_values["mean_symmetry"]
            st.session_state.mean_fractal_dimension = default_values["mean_fractal_dimension"]
        else:
            st.session_state.reset = False

        # Bouton pour lancer le traitement des données
        if st.button("Diagnostic"):
            if all(value == 0 for value in [mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]):
                st.warning("## Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
            else:
                my_data = np.array([[mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]])
                my_data_scaled = scaler_cancer_breast.transform(my_data)
                predictions = model_cancer_breast.predict(my_data_scaled)
                if predictions[0] == 1:

                    st.markdown(f"""**:green[Le modèle prédit que le patient n'a pas de cancer du sein.]**""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""**:red[Le modèle prédit que le patient a un risque de développer un cancer du sein.]**""", unsafe_allow_html=True)

# Contenu des boutons
if selected_page == "Accueil":

    st.markdown(
        f"""
        <div style=' display: flex; align-items: center; justify-content: center;'>
                <img style='width: 60vw;' src={link_logo} />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style=' justify-content: center'>
                <h5 style='margin-top: 50px;'>Cette application offre une fonctionnalité qui vous donne la possibilité d'enregistrer les biomarqueurs spécifiques de vos patients. 
                <br><br>Cette collecte de données permet une évaluation approfondie de leur probabilité de développer diverses pathologies, notamment le cancer du sein, les maladies cardio-vasculaires, le diabète, les affections hépatiques et la maladie rénale chronique. 
                <br><br>Grâce à cette analyse détaillée, vous bénéficiez d'une précieuse perspective prédictive pour anticiper et gérer les risques de santé de manière proactive.</h5>
        </div>
        """,
        unsafe_allow_html=True
    )
elif selected_page == "Informations":

       # Première rangée
    st.markdown(
            f"""
                    <h5 style='text-align: center; margin-top: 0px; margin-bottom: 30px;'>Pour plus d'informations sur les maladies, nous vous invitons à cliquer sur les liens suivants :</h5>
            """,
            unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

    with col1:
        st.markdown(
            f"""
            <div style='width: 10vw; display: flex; flex-direction: column; align-items: center;'>
                <a href="https://www.ameli.fr/lille-douai/assure/sante/themes/cancer-sein">
                    <img style='width: 10vw;' src="https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/breast.png?raw=true" />
                    <h5 style='text-align: center; margin-top: 5px;'>Cancer du sein</h5>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.title(" ")  
    with col3:
        st.markdown(
            f"""
            <div style='width: 10vw; display: flex; flex-direction: column; align-items: center;'>
                <a href="https://www.ameli.fr/lille-douai/assure/sante/themes/diabete">
                    <img style='width: 10vw;' src="https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/diabetes.png?raw=true" />
                    <h5 style='text-align: center; margin-top: 5px;'>Diabète</h5>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.title(" ")
    with col5:
        st.markdown(
            f"""
            <div style='width: 10vw; display: flex; flex-direction: column; align-items: center;'>
                <a href="https://www.ameli.fr/lille-douai/assure/sante/themes/risque-cardiovasculaire">
                    <img style='width: 10vw;' src="https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/heart.png?raw=true" />
                    <h5 style='text-align: center; margin-top: 5px;'>Maladies cardiaques</h5>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Deuxième rangée 
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

    with col1:
        st.title(" ")
    with col2:
        st.markdown(
            f"""
            <div style='width: 10vw; display: flex; flex-direction: column; align-items: center;'>
                <a href="https://www.ameli.fr/lille-douai/assure/sante/themes/maladie-renale-chronique">
                    <img style='width: 10vw;' src="https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/kidney.png?raw=true" />
                    <h5 style='text-align: center; margin-top: 5px;'>Maladie rénale chronique</h5>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.title(" ")
    with col4:
        st.markdown(
            f"""
            <div style='width: 10vw; display: flex; flex-direction: column; align-items: center;'>
                <a href="https://www.ameli.fr/lille-douai/assure/sante/themes/cirrhose-foie">
                    <img style='width: 10vw;' src="https://raw.githubusercontent.com/Ju-stASimpleName/Clinical_Data_Insight/main/logos/liver.png?raw=true" />
                    <h5 style='text-align: center; margin-top: 5px;'>Maladies du foie</h5>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.title(" ")        
elif selected_page == "Maladies cardiaques":
    maladies_cardiaques()
elif selected_page == "Maladies du foie":
    maladies_du_foie()
elif selected_page == "Maladie rénale chronique":
    maladie_renale_chronique()
elif selected_page == "Diabète":
    diabete()
elif selected_page == "Cancer du sein":
    cancer_du_sein()
elif selected_page == "Disclaimer":

    text_disclaimer = "Les prédictions générées par l'application ne peuvent en aucun cas se substituer à l'avis d'un professionnel de la santé.<br><br>Leur but est de fournir un support complémentaire lors du processus de prise de décision concernant d'éventuels examens complémentaires et traitements.<br><br>Il est crucial de souligner que les données que vous saisissez sont entièrement anonymisées et ne font l'objet d'aucune conservation, en stricte conformité avec les directives rigoureuses du Règlement Général de Protection des Données (RGPD)."
    st.sidebar.markdown(f'''**[DISCLAIMER]**<br>{text_disclaimer}''', unsafe_allow_html=True)
    st.header("DISCLAIMER")
    st.markdown(
        f"""
        <div style='justify-content: center '>
                <h5 style=' margin-top: 5px;'>{text_disclaimer}</h5>
        </div>
        """,
        unsafe_allow_html=True
    )
