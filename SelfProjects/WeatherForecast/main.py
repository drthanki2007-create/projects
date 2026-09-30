import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("weather_data.csv")

df["tomorrow_temp"] = df["meantempm"].shift(-1)
df = df.dropna()

features = ['meantempm', 'humidity', 'wspdm', 'precipm']
x = df[features]
y = df['tomorrow_temp']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 24)

model = Ridge()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f} Celsius')