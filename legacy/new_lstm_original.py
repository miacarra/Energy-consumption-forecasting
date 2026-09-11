import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Зареждане на исторически данни
file_path = "hypothetical_energy_data.csv"
df = pd.read_csv(file_path)

# Преобразуване на датите в подходящ формат
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Проверка на наличните колони
print(df.head())

# Избиране на входни характеристики и целева променлива
features = ['Temperature', 'Economic_Index']
target = 'Energy_Consumption'

# Проверка за липсващи стойности
df.dropna(inplace=True)

# Нормализиране на входните данни
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features + [target]])

# Преобразуване обратно в DataFrame за по-лесна работа
df_scaled = pd.DataFrame(data_scaled, columns=features + [target], index=df.index)

# Дефиниране на времеви последователности (24 часа назад)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # Входни характеристики (без целевата променлива)
        y.append(data[i + seq_length, -1])     # Целева променлива (консумация)
    return np.array(X), np.array(y)

seq_length = 24  # 24 часа назад за прогнозиране на следващия час
X, y = create_sequences(data_scaled, seq_length)

# Разделяне на тренировъчни и тестови данни
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Създаване на LSTM модела
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Компилиране на модела
model.compile(optimizer='adam', loss='mse')

# Обучение на модела
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Графика за Mean Squared Error (MSE)
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Mean Squared Error (MSE) по време на обучението')
plt.show()

# Прогнозиране
predictions = model.predict(X_test)

# Де-нормализация на резултатите
predictions_actual = scaler.inverse_transform(np.column_stack([predictions, np.zeros((len(predictions), len(features)))]))[:, 0]
y_test_actual = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), len(features)))]))[:, 0]

# Визуализация на прогнозите
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Реални стойности')
plt.plot(predictions_actual, label='Прогнозирани стойности', linestyle='dashed')
plt.xlabel('Часове')
plt.ylabel('Потребление на енергия (kWh)')
plt.legend()
plt.title('Прогнозиране на потреблението на електроенергия с LSTM')
plt.show()

# Намиране на периода на тестовите данни
test_size = int(len(df) * 0.2)  # 20% от общия брой редове
test_dates = df.index[-test_size:]  # Последните 20% от датите

# Начална и крайна дата на тестовите данни
first_test_date = test_dates.min()
last_test_date = test_dates.max()
from pandas import Timestamp
(Timestamp('2022-05-26 00:00:00'), Timestamp('2022-12-30 00:00:00'))
print(f"Начална дата на тестовите данни: {first_test_date}")
print(f"Крайна дата на тестовите данни: {last_test_date}")

import matplotlib.pyplot as plt

# Създаваме масив с реалните дати за тестовите прогнози
test_dates = df.index[-len(y_test):]  # Вземаме последните дати, съответстващи на тестовите данни

# Визуализация с реални дати
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_actual, label='Реални стойности')
plt.plot(test_dates, predictions_actual, label='Прогнозирани стойности', linestyle='dashed')
plt.xlabel('Дата')
plt.ylabel('Потребление на енергия (kWh)')
plt.legend()
plt.title('Прогнозиране на потреблението на електроенергия с LSTM')
plt.xticks(rotation=45)  # Завъртаме датите, за да се виждат по-добре
plt.show()