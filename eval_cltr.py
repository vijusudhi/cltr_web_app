import streamlit as st
import pandas as pd
import re
from github import Github

import sys
sys.path.append('../')
from util import tokenize

@st.cache(allow_output_mutation=True)
def load_df(group):
    inp = pd.read_csv('eval/group_%s_eval.csv' % group, header=None, 
                  names=['q_id', 'query_text', 'document', 'lang'],
                  delimiter=';')
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
st.write("## **Phase I Evaluation**")

def update(is_relevant_opt, rel_words_sel, submit_button):
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


st.sidebar.write('You are at the Phase I Evaluation of the Master thesis on **Explainable Cross-Lingual Text Retrieval** on Automotive domain. To know more about the thesis and this evaluation, click [here](https://vijusudhi.github.io/cltr_web_app/).')
st.sidebar.write('# **Login**')
st.sidebar.write('Please enter your credentials and click **Enter**.')
username = st.sidebar.text_input('Username', '', 
                                 help="Enter a unique name. You may use your first name.")
password = st.sidebar.text_input('Password', '', type="password", 
                                 help="Enter the password as in the email.")
group = st.sidebar.radio('Group', 
                            options=('A', 'B'),
                        help="Select the group as in the email.")
group = group.lower()

if not username or not password or not group:
    st.sidebar.info('Missing one or more of: username, password, group')
    st.sidebar.write('In case of any technical troubles, please drop a mail at viju.sudhi@audi.de')
    st.stop()

github = Github('%s'%password.strip())
try:
    repository = github.get_user().get_repo('cltr_web_app')
    st.sidebar.success('Login successful!')
    st.sidebar.write('In case of any technical troubles, please drop a mail at viju.sudhi@audi.de')
except:
    st.sidebar.error('Wrong password. Please try again!')
    st.sidebar.write('In case of any technical troubles, please drop a mail at viju.sudhi@audi.de')
    st.stop()    


filename = '%s_log.csv' % username
log_files = [file.name for file in repository.get_contents("logs")]
filepath = 'logs/%s' %filename

if filename not in log_files:
    data = 'q_id, query, document, is_relevant, rel_words'
    f = repository.create_file(filepath, "User %s pushing via PyGithub" %username, data)

inp, inp_qids = load_df(group)
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

latest_iteration = st.empty()
my_bar = st.progress(0)
with st.form(key='my_form'):    
    col1, col2 = st.columns([4,20])
    query_text = curr_df.query_text.values[0]
    document = curr_df.document.values[0]

    col1.write('**ID**')
    col2.write(q_id)    
    col1.write('**Search query**')
    col2.write(query_text)
    col1.write('**Search result**')
    col2.write(document)

    is_relevant_opt = st.radio(
        label="Do you think this result should be returned when you search the query?",
        options=('Yes', 'No'),
        help='Please read the search query and the result document. \
        Select "Yes" if you think the document talks about the query and \
        should be returned while a user searches for this query. \
        Select "No" otherwise.')
    
    print(document, curr_df.lang.values[0])
    words = tokenize.get_tokens(document, lang=curr_df.lang.values[0])

    rel_words_sel = st.multiselect(
        label='Keep only the most relevant words.',
        help='The following words are from the result document. The system \
        returned this document when the particular query was searched because \
        of similar words in the query and the document. In the list, keep only \
        the most important words. Remove irrelevant words.', \
        options=words, default=words)
    
    submit_button = st.form_submit_button(label='Submit', 
                                          help='Click to submit your changes')                 
        
if submit_button:
    if is_relevant_opt == 'Yes' and len(rel_words_sel) == 0:
        st.warning('You answered "Yes" but did not select any relevant words.')
    elif is_relevant_opt == 'No' and len(rel_words_sel) != 0:
        st.warning('You answered "No" but have selected relevant words. \
        Click Next to ignore these words and continue.')
        btn_next = st.button(label='Next',
                             help='Click to proceed to next query',
                             on_click=update, 
                             args=(is_relevant_opt, rel_words_sel, submit_button)
                            )                
    else:
        st.info('Your changes are submitted. Click Next to continue.')
        btn_next = st.button(label='Next',
                             help='Click to proceed to next query',
                             on_click=update, 
                             args=(is_relevant_opt, rel_words_sel, submit_button)
                                    )  
latest_iteration.write('Status: %d / %d' %((len(log_qids)-1), len(inp_qids)))
my_bar.progress((len(log_qids)-1)/len(inp_qids))
