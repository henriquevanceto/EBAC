import streamlit         as st
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
from PIL                 import Image

bank_raw = pd.read_csv('..data/input/bank-additional-full.csv', sep=';')

st.set_page_config(
     page_title='Análise de Telemarketing',
     page_icon=r'../img/pngwing.com.png',
     layout='wide',
     initial_sidebar_state='expanded'
)

st.write('# Análise de Telemarketing')
st.markdown("---")

image = Image.open('../img/1691054390911.png')
st.sidebar.image(image)