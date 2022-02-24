import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

st.image('header.png', use_column_width='auto')

st.info(
    """
    This app predicts if a certain customer is likely to stop doing business with a hypothetical bank based on
    attributes about the customer and their account usage.
    """
)

st.sidebar.title('What is Customer Churn?')
st.sidebar.image('graphic.png')
st.sidebar.info(
    """
    Customer churn is defined as the loss of customers. Using machine learning and historical data, service providers
    could identify customers most likely to stop using a service beforehand and selected for targeted marketing so as to
    perhaps inspire them to stay. 
    """
)

age = st.number_input('Age', step=1.0)
num_products = st.number_input('Number of bank products/services used', step=1.0)
cr_score = st.number_input('Credit score', step=1.0)
country = st.selectbox('Country of residence', ['Germany', 'France', 'Spain'])
acc_status = st.selectbox('Account status', ['Active', 'Inactive'])

if acc_status == 'Active':
    acc_status = 1
else:
    acc_status = 0

process = st.button('Process')


def load_pipeline():
    pipeline = joblib.load('customer_churn.pkl')
    return pipeline


def make_df():
    df = pd.DataFrame({'Age': age,
                       'NumOfProducts': num_products,
                       'CreditScore': cr_score,
                       'Geography': country,
                       'IsActiveMember': acc_status},
                      index=[0])
    return df


def make_classification():
    df = make_df()
    pipeline = load_pipeline()
    prediction = pipeline.predict(df)
    prediction = prediction[0]

    if prediction == 0:
        return st.success('This customer is not likely to terminate his/her account.')
    else:
        return st.error('This customer is likely to terminate his/her account.')


def custom_sum(array_to_sum, num):
    """This function will sum the first 3 elements of each array in a nested array"""
    #  Selecting the second set of values
    arr = array_to_sum[1]
    #  Converting array to list
    arr = arr.tolist()
    #  Summation and creation of new list
    arr[num] = np.array((arr[num][0]+arr[num][1]+arr[num][2],
                         arr[num][3], arr[num][4], arr[num][5], arr[num][6]))
    return arr[num]


def st_shap(plot, height=None):
    shap_html = f'<head>{shap.getjs()}</head><body>{plot.html()}</body>'
    components.html(shap_html, height=height)


def plot_shap():
    #  loading pipeline
    pipeline = load_pipeline()
    #  reading mask
    mask = pd.read_csv('churn_mask.csv')
    #  Creating object to calculate shap values
    explainer = shap.TreeExplainer(pipeline[-1], pipeline[0].transform(mask))
    #  Calculating shap values
    shap_vals = explainer.shap_values(pipeline[0].transform(make_df()))
    shap_vals = custom_sum(shap_vals, 0)
    #  plotting
    return shap.force_plot(explainer.expected_value[1], shap_vals, make_df())


def explain():
    return st.info(
        """
        The plot above provides an insight as to why the underlying model as classified this customer as such.
        The customer's likelihood to terminate his/her account is highlighted in bold text as a probability with a value
        greater than 0.5 (50%) being a pointer to likely account termination.
        
        Attributes are color coded in the plot with red attributes being those that drive up the likelihood of
        termination, while blue attributes are those which drive down the likelihood of termination. The length of each
        attribute quantifies its contribution to the predicted class.
        
        ---
        ###### Key:
        * Age - Age
        * Number of bank products - NumOfProducts
        * Credit Score - CreditScore
        * Country of Residence - Geography
        * Account Status - IsActiveMember (Yes - 1, No - 0)
        """
    )


def output():
    if process:
        st.write('Output:')
        make_classification()
        st_shap(plot_shap())
        st.write('Explanation:')
        explain()
    pass


output()

