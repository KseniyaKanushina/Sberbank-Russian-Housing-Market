pip install streamlit
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib


st.header("Sberbank house prediction")

st.write('My prediction app')

full_sq = st.number_input("full_sq")

life_sq = st.number_input("life_sq")

floor = st.number_input("floor")

max_floor = st.number_input("max_floor")

material = st.number_input("material")

build_year = st.number_input("build_year")

num_room = st.number_input("num_room")

if st.button("Submit"):

    clf = joblib.load("model.pkl")

    X = pd.DataFrame([[full_sq, life_sq, floor, max_floor, material, build_year, num_room]],
                     columns = ["full_sq", "life_sq", "floor", "max_floor", "material", "build_year", "num_room"])

    X = xgb.DMatrix(X)

    prediction = clf.predict(X)[0]


    st.text(f"This instance is a {prediction}")
