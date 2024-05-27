import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv('universal_top_spotify_songs.csv')
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    df['country'] = df['country'].fillna("Global")
    df = df.dropna()
    df = df.sample(n=100, random_state=73)
    df = df.drop(columns=['daily_rank', 'daily_movement', 'weekly_movement', 'snapshot_date', 'spotify_id'])
    df = df.drop(columns=['album_name', 'artists', 'name', 'album_release_date'])
    country_label_encoder = LabelEncoder()
    df['country'] = country_label_encoder.fit_transform(df['country'])
    df_scaler = MinMaxScaler()
    cols_to_scale = ['popularity', 'duration_ms', 'loudness', 'tempo']
    for col in cols_to_scale:
        if str(df[col].dtype) not in ['int64', 'float64']:
            continue
        df[[col]] = df_scaler.fit_transform(df[[col]])
    return df

def split_data(df):
    return train_test_split(df.drop(columns='popularity'), df['popularity'], test_size=0.25, random_state=73)

def get_model_pred(x_train, y_train, x_test, mss, msl, ne):
    rfr = RandomForestRegressor(min_samples_split=mss, n_estimators=ne, min_samples_leaf=msl, n_jobs=-1)
    rfr.fit(x_train, y_train)
    return rfr.predict(x_test)

def draw_graph(x_train, x_test, y_train, y_test, mss, msl):
    x_line = []
    y_line = []
    for i in range (1, 100, 5):
        pred  = get_model_pred(x_train, y_train, x_test, mss, msl, i)
        sc = mean_absolute_error(pred, y_test)
        x_line.append(i)
        y_line.append(sc)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x_line, y_line)
    st.pyplot(fig)

st.subheader('Метод случайного леса')

data = load_data()
data = preprocess_data(data)
x_train, x_test, y_train, y_test = split_data(data)

mss_slider = st.sidebar.slider('mean samples split: ', min_value = 1, max_value = 10, value=2, step=1)
msl_slider = st.sidebar.slider('mean_samples_leaf: ', min_value = 1, max_value = 10, value = 1, step = 1)

draw_graph(x_train, x_test, y_train, y_test, mss_slider, msl_slider)


