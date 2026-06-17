import sys
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Automatically find the root directory and add it to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import data_cleaning

base_dir = os.path.dirname(os.path.abspath(__file__))

df = data_cleaning()

with st.container(border=True):
    st.subheader("Survival Rate by Passenger Class", text_alignment='center')
    pclass_df = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
    pclass_df['Survival Status'] = pclass_df['Survived'].map({0: 'Died', 1: 'Survived'})
    st.bar_chart(
        pclass_df,
        x="Pclass",
        y="Count",
        color="Survival Status"
    )
    st.caption(":rainbow[First class passengers had a much higher survival rate than second or third class.]", text_alignment='center')

with st.container(border=True, gap='small'):
    st.subheader('Survival rate by sex', text_alignment='center')
    sex_df = df.groupby(["Sex", "Survived"]).size().reset_index(name='Count')
    sex_df['Survival Status'] = sex_df['Survived'].map({0: 'Died', 1: 'Survived'})
    sex_df['Sex'] = sex_df['Sex'].map({1: 'Males', 0: 'Female'})
    st.bar_chart(
        sex_df,
        x="Sex",
        y="Count",
        color="Survival Status"
    )
    st.caption(":rainbow[Women were more likely to survive the titanic sex and survival has the most positive correlation]", text_alignment='center')

with st.container(border=True):
    st.subheader('Survival rate by place of embarkment', text_alignment='center')
    raw_df = pd.read_csv(os.path.join(base_dir, '..', 'TITANIC.csv'))  # fixed path
    emb_df = raw_df.groupby(['Embarked', 'Survived']).size().reset_index(name='Count')
    emb_df['Survival Status'] = emb_df['Survived'].map({0: 'Died', 1: 'Survived'})
    st.bar_chart(emb_df, x="Embarked", y="Count", color="Survival Status")
    st.caption("C = Cherbourg, Q = Queenstown, S = Southampton")
    st.caption(":rainbow[The Passengers embarking from Southhampton were generally wealthier which correlated with their survival]",
