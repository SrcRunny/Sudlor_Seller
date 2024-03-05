import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


model = None  # Declare model as a global variable

# Load the model and scaler
try:
    model = load_model('house_price_model.h5')

except FileNotFoundError:
    st.error("Model file not found. Make sure the file path is correct.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

data = pd.read_csv("train.csv")

def main():
    st.title("House Price Prediction")
    st.markdown("(refer from 79 peoples in residential homes in Ames, Iowa)")
    st.image("https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png", width=700)
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    
    if overall_qual == 1:
        st.info("""
            Very Poor quality.
        """)
    elif overall_qual == 2:
        st.info("""
           Poor quality.
        """)
    elif overall_qual == 3:
        st.info("""
        Fair quality.
    """)
    elif overall_qual == 4:
        st.info("""
        Below Average quality.
    """)
    elif overall_qual == 5:
        st.info("""
        Average quality.
    """)
    elif overall_qual == 6:
        st.info("""
        Above Average quality.
    """)
    elif overall_qual == 7:
        st.info("""
        indicates Good quality.
    """)
    elif overall_qual == 8:
        st.info("""
        Very Good quality.
    """)
    elif overall_qual == 9:
        st.info("""
        Excellent quality.
    """)
    elif overall_qual == 10:
        st.info("""
        Very Excellent quality.
    """)
    if st.button("Learn about Overall Quality"):
        st.info("""
            OverallQual is assessed by considering the overall features and condition of the house, 
            including building materials, layout, completion, and the overall quality of all construction. 
            Analyzing OverallQual is crucial in assessing the value of homes in different regions 
            and can provide quick and clear insights for buyers or sellers about the overall quality of a property.
        """)

    grliv_area = st.slider("Above Ground Living Area (sq ft)", 500, 6000, 1500)
    garage_cars = st.slider("Number of Garage Cars", 0, 4, 2)
    full_bath = st.slider("Number of Full Bathrooms", 1, 4, 2)

    future_year = st.number_input("Future Year for Prediction", min_value=2023, max_value=2050, step=1, value=2023)

    if st.button("Predict"):
        try:
            # Wrap the prediction in a try-except block
            input_data = np.array([[overall_qual, grliv_area, garage_cars, full_bath, future_year]])
            scaled_input = scaler.transform(input_data)  # Scale input features

            # Print statements for debugging
            print(f"Scaled Input Shape: {scaled_input.shape}")
            print(f"Model Input Shape: {model.input_shape}")

            prediction = model.predict(scaled_input)

            # Print statements for debugging
            print(f"Prediction Shape: {prediction.shape}")
            print(f"Predicted Price (Before Formatting): {prediction[0][0]}")

            # Access the first element of the prediction array before formatting
            predicted_price = prediction[0][0]

            st.success(f"The predicted house price in {future_year} is: ${'{:,.2f}'.format(predicted_price)}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            # Print the traceback for further analysis
            import traceback
            traceback.print_exc()

        visualize_data(data, overall_qual, grliv_area, garage_cars, full_bath)

def visualize_data(data, overall_qual, grliv_area, garage_cars, full_bath):
    st.subheader("Data Visualization")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    sns.boxplot(x='OverallQual', y='SalePrice', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Overall Quality vs. Sale Price')

    sns.scatterplot(x='GrLivArea', y='SalePrice', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Above Ground Living Area vs. Sale Price')

    sns.boxplot(x='GarageCars', y='SalePrice', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Number of Garage Cars vs. Sale Price')

    sns.boxplot(x='FullBath', y='SalePrice', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Number of Full Bathrooms vs. Sale Price')

    plt.tight_layout()

    st.pyplot(fig)

    st.write("Average House Price Over the Years")
    avg_price_by_year = data.groupby('YearBuilt')['SalePrice'].mean()

    plt.figure(figsize=(12, 6))
    avg_price_by_year.plot(marker='o', linestyle='-', color='b')
    plt.title('Average House Price Over the Years')
    plt.xlabel('Year Built')
    plt.ylabel('Average Sale Price')
    plt.grid(True)
    st.pyplot(plt)

if __name__ == "__main__":
    main()
