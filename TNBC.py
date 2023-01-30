import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from joblib import dump, load

#应用主题
st.set_page_config(
    page_title="ML Medicine",
    #page_icon="🐇",
)
#应用标题
st.title('Machine Learning Application for Predicting Bone Metastasis in TNBC')



# conf
col1, col2, col3 = st.columns(3)
Age = col1.selectbox("Age",('<60','>=60'))
Race = col2.selectbox("Race",('American Indian/Alaska Native',
                              'Asian or Pacific Islander',
                              'Black',
                              'White'),3)
Grade = col3.selectbox("Grade",('Grade Ⅰ(Well differentiated)',
                                'Grade Ⅱ(Moderately differentiated)',
                                'Grade Ⅲ(Poorly differentiated)',
                                'Grade Ⅳ(Undifferentiated)'),1)
Marital_status = col1.selectbox("Marital status",('Unmarried','Married'))
T = col2.selectbox("T stage",('T1','T2','T3','T4'))
#NSE = col2.number_input("NSE (ng/mL)",step=0.01,format="%.2f",value=1.45)
N = col3.selectbox("N stage",('N0','N1','N2', 'N3'))
Brain_mets = col1.selectbox("Brain metastasis",('No','Yes'))
Lung_mets = col2.selectbox("Lung metastasis",('No','Yes'))
Liver_mets = col3.selectbox("Liver metastasis",('No','Yes'))


# str_to_
map = {'<60':0,'>=60':1,
       'American Indian/Alaska Native':0,
       'Asian or Pacific Islander':1,
       'Black':2,
       'White':3,
       'Grade Ⅰ(Well differentiated)':0,
       'Grade Ⅱ(Moderately differentiated)':1,
       'Grade Ⅲ(Poorly differentiated)':2,
       'Grade Ⅳ(Undifferentiated)':3,
       'Unmarried':0,'Married':1,
       'Left':0,'Right':1,'Bilateral':2,
       'T1':0,'T2':1,'T3':2,'T4':3,
       'N0':0,'N1':1,'N2':2,'N3':3,
       'No':0,'Yes':1}

T = map[T]
N =map[N]
Brain_mets = map[Brain_mets]
Lung_mets =map[Lung_mets]
Liver_mets = map[Liver_mets]
Age = map[Age]
Race = map[Race]
Grade = map[Grade]
Marital_status = map[Marital_status]

# 数据读取，特征标注
features = ['Age',  'Race', 'Grade','Marital_status','T','N','Brain_mets', 'Liver_mets', 'Lung_mets',]
target='BM'

#读之前存储的模型
XGB=load( 'XGB.pkl')


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[Age,  Race, Grade,Marital_status,T,N,Brain_mets, Liver_mets, Lung_mets]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Age,  Race, Grade,Marital_status,T,N,Brain_mets, Liver_mets, Lung_mets]]))[0][1])*1000//1/10

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')

