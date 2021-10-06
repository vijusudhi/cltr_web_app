import streamlit as st
import pandas as pd
import re
from github import Github

import sys
sys.path.append('../')
from util import tokenize

@st.cache(allow_output_mutation=True)
def load_df():
    inp = pd.read_csv('input.csv', header=None, 
                  names=['q_id', 'query_text', 'document'])
    inp_qids = inp.q_id.to_list()
    return inp, inp_qids

@st.cache
def convert_df(df):
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
password = st.text_input('Please enter your password', '')

if not username or not password:
    st.info('You have not enetered a username and/or password')
    st.stop()
    
github = Github('%s'%password.strip())
try:
    repository = github.get_user().get_repo('cltr_web_app')
except:
    st.error('Wrong password. Please try again!')
print(repository)

filename = '%s_log.csv' % username
log_files = [file.name for file in repository.get_contents("logs")]
filepath = 'logs/%s' %filename
                
if filename not in log_files:
    data = 'q_id, query, document, is_relevant, rel_words'
    f = repository.create_file(filepath, "User %s pushing via PyGithub" %username, data)

inp, inp_qids = load_df()
file = repository.get_contents(filepath)
lines = file.decoded_content.decode()
log_qids = [line.split(',')[0] for line in lines.split('\n')]
to_do = [qid for qid in inp_qids if qid not in log_qids]

if len(to_do) >= 1:
    q_id = to_do[0]
else:
    st.balloons()
    st.write('You have evaluated all the documents. Thanks for your time! :smile:')
    # st.download_button(
    #     label="Download data as CSV",
    #     data=convert_df(log),
    #     file_name=filename,
    # )      
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

    words = tokenize.get_tokens(document, lang='en')
    
    rel_words_sel = st.multiselect(
        'Remove the irrelevant words',
        words, words)
    
    submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:        
        file = repository.get_contents(filepath)
        prev_data = file.decoded_content.decode()
        data = '%s\n%s, %s, %s, %s, %s'\
                %(prev_data,
                  q_id,
                  query_text,
                  document,
                  is_relevant_opt,
                  ': '.join(rel_words_sel)
                 )
        f = repository.update_file(filepath, 
                                   "User %s updating via PyGithub" %username, 
                                   data, sha=file.sha)         
        
# st.download_button(
#     label="Download data as CSV",
#     data=convert_df(log),
#     file_name='%s_evaluation.csv'%username,
# )  