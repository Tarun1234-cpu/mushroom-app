import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the mushroom dataset CSV (replace with your file path)
# Example dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
# You should download and place locally as 'mushrooms.csv'
df = pd.read_csv('mushrooms.csv', header=None)

# Assign column names (based on UCI Mushroom dataset)
df.columns = [
    'class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
    'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
    'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
    'stalk_surface_below_ring', 'stalk_color_above_ring',
    'stalk_color_below_ring', 'veil_type', 'veil_color',
    'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat'
]

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']  # 'e' = edible, 'p' = poisonous

# Encode categorical features using LabelEncoder for each column
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, 'mushroom_model.joblib')
# Save label encoders including target encoder in one dictionary
label_encoders['class'] = target_le
joblib.dump(label_encoders, 'label_encoders.joblib')

print("Model and label encoders saved!")
