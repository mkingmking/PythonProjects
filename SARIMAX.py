import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Excel dosyasının yolunu değiştirin
excel_file_path = '1429vers.xlsx'

# Pandas ile Excel dosyasının ilk 2000 satırını okuyun
df = pd.read_excel(excel_file_path, nrows=20000)

# Preprocess the data
df['Miktar'] = pd.to_numeric(df['Miktar'].replace(',', '.', regex=True), errors='coerce')
df['Tarih'] = pd.to_datetime(df['Tarih'])
df.set_index('Tarih', inplace=True)

# Aggregate total sales for each day
total_daily_sales = df['Miktar'].resample('D').sum()

print("toplam:::::::::::::::::::::::::::::::::::::::..")
print(total_daily_sales)

# Define the SARIMA model parameters
p = 2  # Autoregressive order
d = 1  # Differencing order
q = 2  # Moving average order
P = 1  # Seasonal autoregressive order
D = 1  # Seasonal differencing order
Q = 1  # Seasonal moving average order
s = 12 # Seasonal period (e.g., 12 for monthly data)

# Fit the SARIMA model
model = SARIMAX(total_daily_sales, order=(p, d, q), seasonal_order=(P, D, Q, s),
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()

# Generate forecast for the next 30 days
forecast = results.get_forecast(steps=30)
forecast_df = forecast.summary_frame()

# Plot the actual total daily sales and forecasted total daily sales
plt.figure(figsize=(10, 5))
plt.plot(total_daily_sales.index, total_daily_sales, label='Actual Total Daily Sales', color='blue')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecasted Total Daily Sales', color='red')
plt.fill_between(forecast_df.index,
                 forecast_df['mean_ci_lower'],
                 forecast_df['mean_ci_upper'], color='pink', alpha=0.5)
plt.title('Total Daily Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
