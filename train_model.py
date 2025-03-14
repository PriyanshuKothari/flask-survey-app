import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# load the dataset
df=pd.read_excel("dummy_npi_data.xlsx" , sheet_name= "Dataset")

df["Login Hour"]= pd.to_datetime(df["Login Time"]).dt.hour

le_region= LabelEncoder()
df["Region"]= le_region.fit_transform(df["Region"])

le_specialty = LabelEncoder()
df["Speciality"] = le_specialty.fit_transform(df["Speciality"])

# Define Target Variable (1 = Likely to Attend, 0 = Unlikely)
df["Likely to Attend"]= (df["Count of Survey Attempts"] > 3).astype(int)

# select Features for training
X= df[["Login Hour", "Usage Time (mins)", "Region", "Speciality", "Count of Survey Attempts"]]
y= df["Likely to Attend"]

# Split into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model & Encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_region, "le_region.pkl")
joblib.dump(le_specialty, "le_specialty.pkl")

print("✅ Model training complete! Model saved as model.pkl")