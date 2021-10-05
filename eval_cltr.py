import streamlit as st
import pandas as pd
import re

import os
cwd = os.getcwd()

print('cwd', cwd)
path = '%s/eval.xlsx' % cwd


col1, mid, col2 = st.columns([3,1,20])
with col1:
    st.image('multilingual-icon-9.jpg', width=100)
with col2:
    st.write("""
    # Explainable Cross-lingual Text Retrieval on Automotive domain
    ## Evaluation """)
    

df = pd.read_excel(path, sheet_name='eval', index_col=0)
df['rel_words'] = df['rel_words'].astype(str)
print(df.keys())

# st.write(df)

to_do = df.index[df['is_relevant'] == 'NM'].tolist()
if len(to_do) >= 1:
    q_id = to_do[0]
else:
    st.balloons()
    st.stop()
    

curr_df = df.loc[q_id]

with st.form(key='my_form'):    
    col1, col2 = st.columns([4,20])
    col1.write('**Query**')
    col2.write(curr_df.query_text)
    col1.write('**Document**')
    col2.write(curr_df.document)
    
    is_relevant_opt = st.radio(
        "Do you think this document is relevant?",
        ('Yes', 'No'))  

    doc = curr_df.document.lower()
    doc = re.sub(r'[!"#$%&\()*+/<=>?@\[\\\\\]^_`{|}~-]', '', doc)
    doc = re.sub(r'\.', '', doc)
    doc = re.sub("'", '', doc)
    words = list(set(doc.split(' ')))
    rel_words_sel = st.multiselect(
        'Pick the relevant words',
        words,
        [words[0], words[1]])
    
    submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        with pd.ExcelWriter(path,
                            mode='a+') as writer:
            df.at[q_id,'is_relevant'] = is_relevant_opt
            df.at[q_id,'rel_words'] = ', '.join(rel_words_sel)
            df.to_excel(writer, sheet_name='eval')
