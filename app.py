import streamlit as st#user interface
import joblib#load the model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns#visulaization


model = joblib.load('insurance_charges_predictor.pkl')#this is uded to load the model


st.title("Insurance Charges Predictor")

# User input fields
import streamlit as st

# Input for Age
age = st.number_input("Age", min_value=18, max_value=100, value=18)

# Input for BMI
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=20.0)

# Input for Number of Children
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

# Display the selected values
st.write(f"Selected Age: {age}")
st.write(f"Selected BMI: {bmi}")
st.write(f"Number of Children: {children}")

gender = st.selectbox("Gender", ("Male", "Female"))
smoker = st.selectbox("Smoker", ("Yes", "No"))
region = st.selectbox("Region", ("Southwest", "Southeast", "Northwest", "Northeast"))


gender = 0 if gender == "Male" else 1
smoker = 1 if smoker == "Yes" else 0


region_encoded = [0, 0, 0, 0]  
if region == "Southwest":
    region_encoded[0] = 1
elif region == "Southeast":
    region_encoded[1] = 1
elif region == "Northwest":
    region_encoded[2] = 1
elif region == "Northeast":
    region_encoded[3] = 1

region_value = region_encoded.index(1) 
input_data = np.array([[age, bmi, children, gender, smoker, region_value]])#input data
predicted_charges = model.predict(input_data)

# Set minimum value to 0
if(predicted_charges<0):
    predicted_charges = -(predicted_charges)
else:
    predicted_charges=predicted_charges


st.write("### Estimated Insurance Charges: ${:,.2f}".format(predicted_charges[0]))#display the prediction

st.write("### Feature Impact Visualization")


features = ['Age', 'BMI', 'Children', 'Gender', 'Smoker', 'Region']#creating a bar plot for visulaization
input_values = [age, bmi, children, gender, smoker, region_value]

plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=input_values, palette='viridis')
plt.title('User Input Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
st.pyplot(plt)

age_range = np.arange(18, 101, 1)#line plot insurance vs charges
bmi_fixed = bmi  # Use the user's BMI for a clearer visualization
charges = []

for a in age_range:
    input_data = np.array([[a, bmi_fixed, children, gender, smoker, region_value]])
    charges.append(model.predict(input_data)[0]) 

plt.figure(figsize=(12, 6))
plt.plot(age_range, charges, label='Insurance Charges', color='blue')
plt.title('Trend of Insurance Charges by Age (Fixed BMI)')
plt.xlabel('Age')
plt.ylabel('Estimated Charges')
plt.axvline(x=age, color='red', linestyle='--', label='Selected Age')
plt.axhline(y=predicted_charges[0], color='green', linestyle='--', label='Predicted Charges')
plt.legend()
st.pyplot(plt)


st.write("### Note:")
st.write("This tool provides an estimate based on user input and a trained model.")
