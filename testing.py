"""
# Making more UI friendly and colors
# Exception handling
"""




import streamlit as st
import pandas as pd
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.beta_set_page_config(page_title="Machine Learning Model",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded")
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def getKeysByValues(dictOfElements, listOfValues):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] in listOfValues:
            listOfKeys.append(item[0])
    return  listOfKeys 

def main():
        
    st.title("Prediction Model")

    task= ["Single Output Model", "Multiple Output Model"]

    choice= st.sidebar.selectbox("Select Task", task)
    if st.button('info'):
        st.info("SFLDSJ")

    if choice=="Single Output Model":

        st.subheader("Single Output Model")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        input_buffer = st.file_uploader("Upload a file(xlsx Supported)", type=("xlsx"))
        if input_buffer is not None:

            data_load_state = st.text('Loading data...')
            data=pd.read_excel(input_buffer)
            data_load_state.text('Loading data...done!')
            column_names= sorted(list(data.columns), key=str.lower)
            column_selected= st.multiselect('Select Column', column_names)
    
            if len(column_selected)!=0:
                    
                data= data[column_selected]
                data.fillna('Unknown', inplace=True)
                st.write(data)
                #st.write(data.dtypes)
                raw_text_1= st.selectbox("Select the Prediction Column", sorted(column_selected, key=str.lower))
                raw_text_2 =st.text_area("Enter the Predicted Variable(Comma Seprated)")
                
                if st.button("Predict"):
                    result=ml_model(data, raw_text_1, raw_text_2, 1)
                    #st.write(result)
            

        
    elif choice=="Multiple Output Model":

        st.subheader("Multiple Output Model")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        input_buffer = st.file_uploader("Upload a file", type=("xlsx"))
        if input_buffer is not None:

            data_load_state = st.text('Loading data...')
            data=pd.read_excel(input_buffer)
            data_load_state.text('Loading data...done!')
            column_names= sorted(list(data.columns), key=str.lower)
            column_selected= st.multiselect('Select Column', column_names)
            data= data[column_selected]
            st.text('Data')
            st.write(data)
            raw_text_1=st.multiselect("Select the Prediction Columns", sorted(column_selected, key=str.lower)) 
            raw_text_2 =st.text_area("Enter the Predicted Variable(Comma Seprated)")
            
            if st.button("Predict"):
                result=ml_model(data, raw_text_1, raw_text_2, 2)

def ml_model(data, input_1, input_2, model_select):
    
    df= data
    df.fillna('Unknown', inplace=True)
    le= LabelEncoder()
    df1= df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
    if model_select==1:

        single_model(df, df1, input_1, input_2)
    
    elif model_select==2:

        multi_model(df, df1, input_1, input_2)

def single_model(og_df, df, input_1, input_2):

    input_1 = input_1.split(' ')
    X= np.array(df.drop([input_1[0]], axis=1))
    Y= np.array(df[input_1[0]])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)
    
    clf=RandomForestClassifier(n_estimators=2)
    clf.fit(X_train, Y_train)
    pred= clf.predict(X_test)
    acc= accuracy_score(pred, Y_test)
    res= "{:.0%}".format(acc)
    st.write('''
    #### Confidence Level:
    ''')
    st.write(f'<font color="#f63366"> {res}</font>', unsafe_allow_html=True)
    #st.write('<font color="red">"Confidence Level: {:.0%}".format(acc)</font>', unsafe_allow_html=True)
        
    test_model(og_df, clf, input_1, input_2)

def multi_model(og_df, df, input_1, input_2):

    input_1= ','.join(input_1)
    input_1 = input_1.split(',')
    input_1 = [(x.strip()) for x in input_1]
    X= np.array(df.drop([x for x in input_1], axis=1))
    Y= np.array(df[[x for x in input_1]])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)

    clf= MultiOutputClassifier(RandomForestClassifier(n_estimators=2), n_jobs=-1)
    clf.fit(X_train, Y_train)
    pred= clf.predict(X_test)
    c= pred==Y_test
    acc= np.count_nonzero(c)/(2*len(X_test))
    res= "{:.0%}".format(acc)
    st.write('''
    #### Confidence Level:
    ''')
    st.write(f'<font color="#f63366"> {res}</font>', unsafe_allow_html=True)
    
    test_model(og_df, clf, input_1, input_2)

def test_model(df, clf, input_1, input_2):

    try:
        le= LabelEncoder()
        final_dict={}
        for col in df.columns:
            le.fit(df[col])
            final_dict={**final_dict, **dict(zip(le.classes_, le.transform(le.classes_)))}
        
        variable=[]

        input_2 = input_2.split(',')
        variable = [(x.strip()) for x in input_2]
            
        number_list=[]
        
        for i in range(len(variable)):
            number_list.append(final_dict.get(variable[i]))
        
        pred=clf.predict([number_list])
        pred= pred.flatten()
        answer=[]
        for i in range(len(pred)):
            le.fit(df[input_1[i]])
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            answer.append(getKeysByValues(le_name_mapping, pred))
            pred=np.delete(pred, 0)
        st.write("""
        #### Predicted Variable: 
        """)
        for i in answer:
            st.write(f'<font color="#f63366"> {i[0]} </font>', end = ' ', unsafe_allow_html=True)
        #st.write(answer)
    
    except ValueError:
        st.error("Inavlid Keyword! Try again")

main()