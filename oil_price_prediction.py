import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Step 1: Fetch historical data
print("Fetching historical data...")
data = yf.download("CL=F", start="2020-01-01", end="2023-12-31")

# Flatten the MultiIndex columns
data.columns = ['_'.join(col) for col in data.columns]

# Print the fetched data and its columns
print("Fetched data:")
print(data.head())
print("Columns in the fetched data:")
print(data.columns)

# Reset the index to make 'Date' a column
data = data.reset_index()

# Save data to a CSV file
data.to_csv("data/crude_oil_prices.csv")

# Step 2: Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv("data/crude_oil_prices.csv")

# Ensure 'Date' is in the correct format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Create a new column for the target variable (price after 7 days)
df['Target'] = df['Close_CL=F'].shift(-7)  # Predict price 7 days in the future

# Drop rows with missing values
df = df.dropna()

# Step 3: Train the model
print("Training the model...")
X = df[['Close_CL=F', 'High_CL=F', 'Low_CL=F', 'Open_CL=F', 'Volume_CL=F']]  # Added 'Open_CL=F'
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # Tuned hyperparameters
model.fit(X_train, y_train)

# Step 4: Make predictions
print("Making predictions...")
predictions = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Step 6: Visualize results
print("Plotting results...")
plt.figure(figsize=(10, 6))
plt.plot(df['Date'][-len(y_test):], y_test, label="Actual Prices", color='blue')
plt.plot(df['Date'][-len(y_test):], predictions, label="Predicted Prices", color='red')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Crude Oil Price Prediction")
plt.legend()
plt.savefig("plot.png")  # Save the plot as an image
plt.show()

# Step 7: Save the model
print("Saving the model...")
joblib.dump(model, "models/best_oil_price_predictor.pkl")  # Updated file name
print("Done!")