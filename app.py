import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_classifier.pkl')

def main():
    st.title('Loan Approval Prediction')
    st.write('Fill in the details below to get the loan approval prediction:')

    # Load the dataset
    df = pd.read_csv('dataset.csv')

    # Input fields for user to enter data
    no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
    education = st.selectbox('Education', df['education'].unique())
    self_employed = st.radio('Self Employed', ['Yes', 'No'])
    income_annum = st.number_input('Annual Income', min_value=float(df['income_annum'].min()), format='%f')
    loan_amount = st.number_input('Loan Amount', min_value=float(df['loan_amount'].min()), format='%f')
    loan_term = st.number_input('Loan Term (Months)', min_value=int(df['loan_term'].min()), format='%d')
    cibil_score = st.number_input('CIBIL Score', min_value=int(df['cibil_score'].min()), max_value=int(df['cibil_score'].max()), step=1)
    residential_assets_value = st.number_input('Residential Assets Value', min_value=float(df['residential_assets_value'].min()), format='%f')
    commercial_assets_value = st.number_input('Commercial Assets Value', min_value=float(df['commercial_assets_value'].min()), format='%f')
    luxury_assets_value = st.number_input('Luxury Assets Value', min_value=float(df['luxury_assets_value'].min()), format='%f')
    bank_asset_value = st.number_input('Bank Asset Value', min_value=float(df['bank_asset_value'].min()), format='%f')

    # Make prediction when 'Predict' button is clicked
    if st.button('Predict'):
        # Create a DataFrame from user input
        input_data = {
            'no_of_dependents': [no_of_dependents],
            'education': [education],
            'self_employed': [1 if self_employed == 'Yes' else 0],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value]
        }
        input_df = pd.DataFrame(input_data)

        # Convert categorical variable (education) to dummy/indicator variables
        input_df = pd.get_dummies(input_df, columns=['education'])

        # Make prediction using the loaded model
        prediction = model.predict(input_df)
        prediction_label = 'Approved' if prediction[0] == 1 else 'Rejected'

        # Display prediction result
        st.success(f'Prediction: Loan Application is {prediction_label}')

if __name__ == '__main__':
    main()
