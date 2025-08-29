import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import tensorflow as tf
import time

from datetime import datetime
from pydantic import BaseModel

st.set_page_config(page_title ='M15T01',
                   layout ='wide')

st.markdown('# Módulo 15, Tarefa 01')
st.markdown('---')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.markdown('---')

st.subheader('Map of all pickups')

st.map(data)

st.markdown('---')

hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)



hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

st.markdown('---')

st.subheader('Raw data')
st.write(data)

st.markdown('---')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.markdown('---')

def load_data(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df

df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
st.dataframe(df)

st.button("Rerun")

st.markdown('---')

class MyCustomClass:
    def __init__(self, initial_score: int):
        self.my_score = initial_score

def hash_func(obj: MyCustomClass) -> int:
    return obj.my_score  # or any other value that uniquely identifies the object

@st.cache_data(hash_funcs={MyCustomClass: hash_func})
def multiply_score(obj: MyCustomClass, multiplier: int) -> int:
    return obj.my_score * multiplier

initial_score = st.number_input("Enter initial score", value=15)

score = MyCustomClass(initial_score)
multiplier = 2

st.write(multiply_score(score, multiplier))

st.markdown('---')

class Person(BaseModel):
    name: str

@st.cache_data
def identity(person: Person):
    return person

person = identity(Person(name="Lee"))
st.write(f"The person is {person.name}")

st.markdown('---')

def load_base_model(option):
    if option == 1:
        return tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
    else:
        return tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")

def load_layers(base_model):
    return [layer.name for layer in base_model.layers]

option = st.radio("Model 1 or 2", [1, 2])

base_model = load_base_model(option)

layers = load_layers(base_model)

st.write(layers)

st.markdown('---')

tz = pytz.timezone("Europe/Berlin")

@st.cache_data
def load_data(dt):
    return dt

now = datetime.now()
st.text(load_data(dt=now))

now_tz = tz.localize(datetime.now())
st.text(load_data(dt=now_tz))

st.markdown('---')

@st.cache_data
def get_data():
    df = pd.DataFrame({"num": [112, 112, 2, 3], "str": ["be", "a", "be", "c"]})
    return df

@st.cache_data
def show_data(data):
    time.sleep(2)  # This makes the function take 2s to run
    return data

df = get_data()
data = df["str"].unique()

st.dataframe(show_data(data))
st.button("Re-run")

st.markdown('---')

@st.cache_data
def get_data():
    df = pd.DataFrame({"num": [112, 112, 2, 3], "str": ["be", "a", "be", "c"]})
    return df

@st.cache_data
def show_data(data):
    time.sleep(2)  # This makes the function take 2s to run
    return data

df = get_data()
data = df["str"].unique()

st.dataframe(show_data(data))
st.button("Re-run")

st.markdown('---')

def show_data():
    st.header("Data analysis")
    data = api.get(...)
    st.success("Fetched data from API!")
    st.write("Here is a plot of the data:")
    st.line_chart(data)
    st.write("And here is the raw data:")
    st.dataframe(data)

st.markdown('---')

def get_data():
    num_rows = st.slider("Number of rows to get")  # 👈 Add a slider
    data = api.get(..., num_rows)
    return data