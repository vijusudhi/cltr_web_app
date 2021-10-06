import streamlit as st
import pandas as pd
import re
import json
from pathlib import Path

import os
cwd = os.getcwd()

@st.cache(allow_output_mutation=True)
def load_df():
    inp = pd.read_csv('input.csv', header=None, 
                  names=['q_id', 'query_text', 'document'])
    inp_qids = inp.q_id.to_list()
    return inp, inp_qids

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False, header=False).encode('utf-8')

col1, mid, col2 = st.columns([3,1,20])
with col1:
    st.image('multilingual-icon-9.jpg', width=100)
with col2:
    st.write("""
    # Explainable Cross-lingual Text Retrieval on Automotive domain
    """)
st.write("## **Evaluation of explanations**")


username = st.text_input('Please enter your username', '')
if not username:
    st.stop()
path = '%s/%s_log.csv' % (cwd, username)
log_file = Path(path)
if not log_file.exists():
    data = {}
    data = pd.DataFrame(data)
    data.to_csv(path, mode='a+', index=False, header=False)

inp, inp_qids = load_df()
log = pd.read_csv(path, header=None, 
                  names=['q_id', 'query_text', 'document', 'is_relevant', 'rel_words'])
log_qids = log.q_id.to_list()
to_do = [qid for qid in inp_qids if qid not in log_qids]

if len(to_do) >= 1:
    q_id = to_do[0]
else:
    st.balloons()
    st.write('You have evaluated all the documents. Thanks for your time! :smile:')
    st.download_button(
        label="Download data as CSV",
        data=convert_df(log),
        file_name='evaluation.csv',
    )      
    st.stop()

curr_df = inp[inp['q_id'] == q_id]  

with st.form(key='my_form'):    
    col1, col2 = st.columns([4,20])
    query_text = curr_df.query_text.values[0]
    document = curr_df.document.values[0]
    
    col1.write('**ID**')
    col2.write(q_id)    
    col1.write('**Query**')
    col2.write(query_text)
    col1.write('**Document**')
    col2.write(document)
    
    is_relevant_opt = st.radio(
        "Do you think this document is relevant?",
        ('Yes', 'No'))  

    doc = document.lower()
    doc = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', '', doc)
    doc = re.sub(r'\.', '', doc)
    doc = re.sub("'", '', doc)
    words = []
    for word in doc.split(' '):
        if word and word not in words:
            words.append(word)
    rel_words_sel = st.multiselect(
        'Pick the relevant words',
        words,
        words)
    
    submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:        
        data = {}
        data['q_id'] = q_id
        data['query_text'] = query_text
        data['document'] = document
        data['is_relevant'] = is_relevant_opt
        data['rel_words'] = [', '.join(rel_words_sel)]
        data = pd.DataFrame(data)
        data.to_csv(path, mode='a+', index=False, header=False)
        
st.download_button(
    label="Download data as CSV",
    data=convert_df(log),
    file_name='%s_evaluation.csv'%username,
)  