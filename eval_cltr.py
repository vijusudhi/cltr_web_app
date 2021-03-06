import streamlit as st
import pandas as pd
import re
from github import Github

import sys
sys.path.append('../')
from util import tokenize

st.set_page_config(layout="centered")

@st.cache(allow_output_mutation=True)
def load_df(group):
    # inp = pd.read_csv('eval/group_%s_eval.csv' % group, header=None, 
    #               names=['q_id', 'query_text', 'document', 'lang'],
    #               delimiter=',')
    inp = pd.read_csv('eval/group_%s_extended.csv' % group)
    inp_qids = inp.q_id.to_list()
    return inp, inp_qids

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


st.sidebar.write('You are at the Phase I Evaluation of the Master thesis on \
**Explainable Cross-Lingual Text Retrieval** on Automotive domain. To know \
more about the thesis and this evaluation, click [here] \
(https://vijusudhi.github.io/cltr_web_app/).')
st.sidebar.write('# **Login**')
st.sidebar.write('Please enter your credentials and click **Enter**.')
username = st.sidebar.text_input('Username', '', 
                    help="Enter a unique username, e.g. your first name.")
pw1 = "ghp"
pw2 = "_RCMyRYqp"
pw3 = "liQMdoKwN"
pw4 = "CALyNMchME"
pw5 = "KBa4XbvhN"
password = f"{pw1}{pw2}{pw3}{pw4}{pw5}"
# password = st.sidebar.text_input('Password', '', type="password", 
#                                  help="Enter the password you received in the email.")
group = st.sidebar.text_input('Group (A / B / C)', 
                        help='Enter the group you received in the email.').lower()

message_field = st.sidebar.empty()
btn_yes_field = st.sidebar.empty()

if not username or not password or not group:
    message_field.info('You have not entered a username or a group. Please try again!')
    st.stop()
    
if '_' in username:
    message_field.info('You can not use an underscore (_) in your username. Please try again!')
    st.stop()    

github = Github('%s'%password.strip())
try:
    repository = github.get_user().get_repo('cltr_web_app')
except:
    message_field.error('Incorrect password. Please check your email.')
    st.stop()  

filename = '%s_%s_log.csv' % (username, group)
log_files = [file.name for file in repository.get_contents("logs")]
usernames = [file.split('_')[0] for file in log_files if file != '.gitignore']
usergroups = [file.split('_')[1] for file in log_files if file != '.gitignore']
filepath = 'logs/%s' % filename  

if username in usernames:
    if 'proceed' in st.session_state:
        if st.session_state.proceed != username:
            message_field.warning('User already found. If this is you, click Yes to proceed.')
            btn_yes = btn_yes_field.button('Yes, proceed')
            if not btn_yes:
                st.stop()
        else:
            pass
    else:
        message_field.warning('User already found. If this is you, click Yes to proceed.')
        btn_yes = btn_yes_field.button('Yes, proceed')
        if not btn_yes:
            st.stop()
    
    usergroup = usergroups[usernames.index(username)]
    if usergroup != group:
        btn_yes_field.empty()
        message_field.warning('You entered Group \'%s\' before. \
        Please enter this group again to continue.' %usergroup.upper())
        st.stop()
else:
    data = 'q_id, query, is_relevant, rel_words'
    f = repository.create_file(filepath, "User %s pushing via PyGithub" %username, data)
    message_field.success('Login successful!')      

btn_yes_field.empty()
message_field.success('Login successful!')
st.session_state.proceed = username
            
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
    st.stop()

curr_df = inp[inp['q_id'] == q_id]  

latest_iteration = st.empty()
my_bar = st.progress(0)
with st.form(key='my_form'):    
    col1, col2, col3 = st.columns([4,15, 15])
    query_text = curr_df.query_text.values[0].strip()
    query_text_tr = curr_df.query_text_tr.values[0].strip()
    document = curr_df.document.values[0].strip()
    document_tr = curr_df.document_tr.values[0].strip()

    col1.write('**ID**')
    if 'ST' in q_id:
        color = '#e0e0e0'
    else:
        color = '#fdc3cd'
    col2.markdown('<span style="%s: %s"><b>%s</b></span>' %('background-color', color, q_id), 
                 unsafe_allow_html=True)
    col3.markdown("")
    with st.container():
        col1, col2, col3 = st.columns([4,15, 15])
        col1.write('**Search query**')
        col2.write(query_text)
        col3.write(f"*{query_text_tr}*")
    with st.container():
        col1, col2, col3 = st.columns([4,15, 15])
        col1.write('**Search result**')
        col2.write(document)
        document_tr = re.sub("\*", "", document_tr)
        col3.write(f"*{document_tr}*")        

    is_relevant_opt = st.radio(
        label="Do you think this result should be returned when you search the given query?",
        options=('Yes', 'No'),
        help='Please read the search query and the search result. \
        Select "Yes" if you think the result talks about the query and \
        should be returned while a user searches for this query. \
        Select "No" otherwise.')
    
    words = tokenize.get_tokens(document, lang=curr_df.lang.values[0])
    words_tr = str(curr_df.keywords_tr.values)
    words_tr = re.sub("[\[\]]", '', words_tr)
    words_tr = [re.sub("[\'\"]", "", word) for word in words_tr.split(",")]
    
    words_combined = []
    for word, word_tr in zip(words, words_tr):
        word = word.strip()
        word_tr = word_tr.strip()
        if word == word_tr:
            comb = word
        else:
            comb = "%s (%s)" % (word, word_tr)
        words_combined.append(comb)
    
    rel_words_sel = st.multiselect(
        label='Keep only the words you think are the most relevant while considering the given query.',
        help='The following words are from the search result. The system \
        could have returned this result because \
        of similar words in the query and the result document. \
        From the list of words below, remove irrelevant words and \
        retain only the most relevant words.', \
        options=words_combined, default=words_combined)
    
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
