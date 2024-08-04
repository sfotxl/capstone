import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from datetime import datetime

# Display a welcome message in the console
print('Welcome to the Western Governors Airlines Air Passenger Number Prediction App!')
print('Here are a few statistics regarding this model...')

# Create a folder for holding visualizations of the model.
# Visualizations will be saved to the visualizations folder in the root directory
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Read the historical data from AirPassengersData
df = pd.read_csv('AirPassengersData.csv', parse_dates=['Month'], index_col='Month')

df.index = df.index.to_period('M').to_timestamp()
df['Month'] = df.index.month
df['Year'] = df.index.year

# Designate features and target variable
X = df[['Month', 'Year']]
y = df['#Passengers']

# Split the data into training + testing data
XTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, shuffle=True)

# Train the model
model = LinearRegression()
model.fit(XTrain, yTrain)

# Create predictions
yPredictions = model.predict(xTest)

# Evaluate the model and share metrics in the console
meanAbsoluteError = mean_absolute_error(yTest, yPredictions)
rootSquaredError = root_mean_squared_error(yTest, yPredictions)
r2Score = r2_score(yTest, yPredictions)

print('Mean Absolute Error of model: ', meanAbsoluteError)
print('Root Mean Squared Error of model: ', rootSquaredError)
print('R2 Score of model: ', r2Score)

# Create a time series plot of historical data - Visualization 1
plt.figure(figsize=(10,6))
plt.plot(df.index, df['#Passengers'], label='Number of Passengers')
plt.title('Monthly Number of Air Passengers at Western Governors Airlines')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.savefig(os.path.join(output_dir, 'HistoricalPassengerData.png'))
plt.show()

# Plot historical data by month - Visualization 2
plt.figure(figsize=(10, 6))
df['Month'] = df.index.month
for month in range(1, 13):
    plt.plot(df[df['Month'] == month].index.year, df[df['Month'] == month]['#Passengers'], marker='o', label=month)
plt.title('Historical Air Passenger Numbers By Month')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend(title='Month')
plt.savefig(os.path.join(output_dir, 'HistoricalDataByMonth.png'))
plt.show()

# Plot the predictions - Visualization 3
plt.figure(figsize=(10, 6))
plt.plot(yTest.index, yTest, marker='o', linestyle='', label='Actual', color='blue')
plt.plot(yTest.index, yPredictions, marker='o', linestyle='', label='Predicted', color='red')
plt.title('Actual vs Predicted Number of Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.savefig(os.path.join(output_dir, 'ActualVsPredicted.png'))
plt.show()

# UI for accepting user input and predictions


def get_date_for_prediction():
    date_str = input('Enter a month (YYYY-MM) for which you would like a prediction for air passenger numbers: ')
    try:
        date = datetime.strptime(date_str, '%Y-%m')
        return pd.DataFrame([[date.month, date.year]], columns=['Month', 'Year'])
    except ValueError:
        print('Invalid date formate. Try again.')
        return get_date_for_prediction()


while True:
    user_input = get_date_for_prediction()
    prediction = model.predict(user_input)
    # Print the prediction for the input date
    print(f'Predicted number of passengers for {user_input.iloc[0,1]}-{user_input.iloc[0,0]:02d} (YYYY-MM): {prediction[0]} passengers')

    cont = input('Do you want to continue (Y/N): ')
    if cont != 'Y':
        break