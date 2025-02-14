import pandas as pd
from pygam import LinearGAM, s, f

# Load training data and convert timestamp to datetime
train_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])

# Extract time-based features
train_data['year'] = train_data['Timestamp'].dt.year
train_data['month'] = train_data['Timestamp'].dt.month
train_data['day'] = train_data['Timestamp'].dt.weekday
train_data['hour'] = train_data['Timestamp'].dt.hour

# Load test data and convert timestamp to datetime
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])

# Extract time-based features
test_data['year'] = test_data['Timestamp'].dt.year
test_data['month'] = test_data['Timestamp'].dt.month
test_data['day'] = test_data['Timestamp'].dt.weekday
test_data['hour'] = test_data['Timestamp'].dt.hour

# Select relevant features
X_train = train_data[['month', 'day', 'hour']]
y_train = train_data['trips']
X_test = test_data[['month', 'day', 'hour']]

# Create and train the model using a Generalized Additive Model (GAM)
gam_model = LinearGAM(s(0) + f(1) + s(2))
gam_model.fit(X_train, y_train)

# Make predictions
predictions = gam_model.predict(X_test)

# Display the first few predictions
print(predictions[:10])
