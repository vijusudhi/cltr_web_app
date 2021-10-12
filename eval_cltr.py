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
                  delimiter=',')
    inp_qids = inp.q_id.to_list()
    return inp, inp_qids

# @st.cache(allow_output_mutation=True)
def load_users():
    users = pd.read_csv('users.csv', header=None, 
            names=['user', 'group'],
            delimiter=',')
    return users

@st.cache
def convert_df(df):
    return df.to_csv(index=False, header=False).encode('utf-8')

col1, mid, col2 = st.columns([3,1,20])
with col1:
    st.image('images/multilingual-icon-9.jpg', width=100)
with col2:
    st.write("""
    # Explainable Cross-lingual Text Retrieval on Automotive domain
    """)
st.write("## **Phase I Evaluation**")

def update(is_relevant_opt, rel_words_sel, submit_button):
    file = repository.get_contents(filepath)
    prev_data = file.decoded_content.decode()
    data = '%s\n%s, %s, %s, %s'\
            %(prev_data,
              q_id,
              query_text,
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
                    help="Enter the username as in the email.")
password = st.sidebar.text_input('Password', '', type="password", 
                                 help="Enter the password as in the email.")
group = st.sidebar.radio('Select the group', 
                        options=('A', 'B', 'C'),
                        help='Select the group as in your email.').lower()
# users_df = load_users()

message_field = st.sidebar.empty()

if not username or not password:
    message_field.info('You have not entered username and/or password')
    st.stop()

# if username not in users_df.user.to_list():
#     message_field.error('Incorrect username. Please check your email.')
#     st.stop()

github = Github('%s'%password.strip())
try:
    repository = github.get_user().get_repo('cltr_web_app')
except:
    message_field.error('Incorrect username. Please check your email.')
    st.stop()  

filename = '%s_%s_log.csv' % (username, group)
log_files = [file.name for file in repository.get_contents("logs")]
print('log_files', log_files)
usernames = [file.split('_')[0] for file in log_files if file != '.gitignore']
usergroups = [file.split('_')[1] for file in log_files if file != '.gitignore']
print('usernames', usernames)
print('usergroups', usernames)
filepath = 'logs/%s' % filename

# if filename not in log_files:
#     data = 'q_id, query, is_relevant, rel_words'
#     f = repository.create_file(filepath, "User %s pushing via PyGithub" %username, data)
#     message_field.success('Login successful!')        

if username in usernames:
    if 'proceed' in st.session_state:
        pass
    else:
        message_field.warning('User already found. If this is you, click Yes.')
        btn_yes = st.sidebar.button('Yes, it is me.')
        if not btn_yes:
            st.stop()
    
    usergroup = usergroups[usernames.index(username)]
    if usergroup != group:
        message_field.warning('You chose Group \'%s\' before. \
        Please choose this group again to continue.' %usergroup.upper())
        st.stop()
else:
    data = 'q_id, query, is_relevant, rel_words'
    f = repository.create_file(filepath, "User %s pushing via PyGithub" %username, data)
    message_field.success('Login successful!')      

if 'proceed' not in st.session_state:
    message_field.success('Login successful!')
    st.session_state.proceed = 'yes'
            
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
    if 'ST' in q_id:
        color = '#e0e0e0'
    else:
        color = '#fdc3cd'
    col2.markdown('<span style="%s: %s"><b>%s</b></span>' %('background-color', color, q_id), 
                 unsafe_allow_html=True)
    col1.write('**Search query**')
    col2.write(query_text)
    col1.write('**Search result**')
    col2.write(document)

    is_relevant_opt = st.radio(
        label="Do you think this result should be returned when you search the given query?",
        options=('Yes', 'No'),
        help='Please read the search query and the search result. \
        Select "Yes" if you think the result talks about the query and \
        should be returned while a user searches for this query. \
        Select "No" otherwise.')
    
    print(document, curr_df.lang.values[0])
    words = tokenize.get_tokens(document, lang=curr_df.lang.values[0])

    rel_words_sel = st.multiselect(
        label='Keep only the words you think are the most relevant while considering the given query.',
        help='The following words are from the search result. The system \
        could have returned this result because \
        of similar words in the query and the result document. \
        From the list of words below, remove irrelevant words and \
        retain only the most relevant words.', \
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
                             help='Click to proceed to next query.',
                             on_click=update, 
                             args=(is_relevant_opt, rel_words_sel, submit_button)
                                    )  
latest_iteration.write('Status: %d / %d' %((len(log_qids)-1), len(inp_qids)))
my_bar.progress((len(log_qids)-1)/len(inp_qids))
