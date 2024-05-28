import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv('universal_top_spotify_songs.csv')
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    df['country'] = df['country'].fillna("Global")
    df = df.dropna()
    df = df.sample(n=500, random_state=73)
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

def draw_graph(x_train, x_test, y_train, y_test, mss, msl, ne_step, ne_range):
    x_line = []
    y_line = []
    for i in range (ne_range[0], ne_range[1], ne_step):
        pred  = get_model_pred(x_train, y_train, x_test, mss, msl, i)
        sc = mean_absolute_error(pred, y_test)
        x_line.append(i)
        y_line.append(sc)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x_line, y_line)
    st.pyplot(fig)

st.subheader('Метод случайного леса')

if st.checkbox('Описание метода'):
    st.markdown("""
Алгоритм случайного леса сочетает в себе две основные идеи: метод бэггинга и метод случайных подпространств.

Случайный лес можно рассматривать как алгоритм бэггинга над решающими деревьями.

Но при этом каждое решающее дерево строится на случайно выбранном подмножестве признаков.
Эта особенность называется "feature bagging" и основана на методе случайных подпространств.

Метод случайных подпространств позволяет снизить коррелированность между деревьями и избежать переобучения. 
Базовые алгоритмы обучаются на случайно выбранных подмножествах признаков.
Ансамбль моделей, использующих метод случайного подпространства, можно построить, используя следующий алгоритм:

Пусть количество объектов для обучения равно $N$, а количество признаков $D$.
Выбирается число отдельных моделей в ансамбле $L$.
Для каждой отдельной модели $l = 1..L$ выбирается число признаков $dl (dl < D)$. Как правило, для всех моделей используется только одно значение
$dl$.

Для каждой отдельной модели $l$ создается обучающая выборка путем отбора $dl$ признаков из $D$.
Производится обучение моделей $l = 1..L$, каждая модель обучается на отдельном подмножестве из $dl$ признаков.
Чтобы применить модель ансамбля к тестовой выборке, объединяются результаты отдельных моделей или мажоритарным голосованием или более сложными способами.
                
---
В конкретно моей НИР модель случайного леса показала наилучшие результаты, поэтому я выбрал её для визуализации.
""")

if st.checkbox('Описание гиперпараметров и метрик'):
    st.markdown("""
            Для оценки качеста модели используется средняя абсолютная погрешность (**mean_absolute_error**).

            Можно изменить гиперпараметры модели:
            - **min_samples_leaf**: определяет количество сэмплов, которое должно быть у ноды-листка
            - **min_samples_split**: определяет количество сэмплов, которое нужно, чтобы разделить внутренний узел.
                
            С учётом этих гиперпараметров на графике рассчитывается качество моделей для главного гиперпараметра `n_estimators` (количнство деревьев).
            """)
    
data = load_data()
data = preprocess_data(data)

if st.checkbox('Корреляционная матрица'):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig)


x_train, x_test, y_train, y_test = split_data(data)

st.sidebar.subheader('Гиперпараметры')
mss_slider = st.sidebar.slider('mean samples split: ', min_value = 1, max_value = 10, value=2, step=1)
msl_slider = st.sidebar.slider('mean_samples_leaf: ', min_value = 1, max_value = 10, value = 1, step = 1)

st.sidebar.subheader('Конфигурация графика')
ne_step = st.sidebar.slider('Шаг при построении графика: ', min_value = 1, max_value = 25, value = 10, step = 1)
ne_range = st.sidebar.slider('Диапазон количетсва деревьев на графике: ', min_value = 1, max_value = 300, value = (1, 100), step = 1)


draw_graph(x_train, x_test, y_train, y_test, mss_slider, msl_slider, ne_step, ne_range)


