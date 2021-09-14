import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.cistem import Cistem
from sentence_transformers import SentenceTransformer
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import datetime
import umap.umap_ as umap
import umap.plot
import numpy as np
import re
import time
import pandas as pd
import textwrap
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import pickle

import sys
sys.path.append('../')
from util import corpus, tokenize, util, alignment, word2vec, retrieval


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

CONT_MODEL_PATH = "model/make-multilingual-sys-2021-07-24_21-03-57"
EUROPARL_DATA_PATH = "model/europarl_df.pkl"
EN_VECTORS_PATH = 'model/en_vectors.pkl'
DE_VECTORS_PATH = 'model/de_vectors.pkl'
PROJ_PATH = 'model/proj_2d.pkl'
NUM_RETRIEVAL = 5

PROJ_LINK = '1xP8dWx7y6EZjaZYQJ1PnRNMbAeJNBYJY'
EN_VECTORS_LINK = '10ktdgB3-BqDWHzIMo468K_QsZQLYESK6'
DE_VECTORS_LINK = '1wM_bZhFUg1VbaA-6E7OTgi9Bj530x9hE'
EUROPARL_DATA_LINK = '1KZJrIetQmuoDKZVzwZxvW6lyjaZkEBO7'
CONT_MODEL_LINK = '1KLx3m_6kDXM3Dss_bgjIRIv0vMKjRBag'

def get_sentence_vector(model, sent):
    sent_vector = model.encode(sent, show_progress_bar=False)
    sent_vector = torch.from_numpy(sent_vector)
    return sent_vector
    
def get_retrieved_docs(query):
    # Compute the target vectors before-hand. These do not change.   
    query_vector = get_sentence_vector(model, [query])
    output = F.cosine_similarity(query_vector, de_vectors, dim=1)
    y_pred_de = torch.topk(output, NUM_RETRIEVAL).indices.numpy()
    
    output = F.cosine_similarity(query_vector, en_vectors, dim=1)
    y_pred_en = torch.topk(output, NUM_RETRIEVAL).indices.numpy()    
    return y_pred_en, y_pred_de

@st.cache
def load_model():

    def get_clusters():
        # Plot UMAP clusters
        df = pd.DataFrame(corpus_en[:5000]+corpus_de[:5000])
        df.columns = ['text']
    
        colors = ['lavender' for k in df['text'].values]
    
        fig_2d = px.scatter(
            proj_2d, x=0, y=1,
            hover_name=df['text'].apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))))
    
        fig_2d.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
        fig_2d.update_traces(hoverlabel=dict(align="left"), marker_color=colors)
        
        return fig_2d, df, colors
    
    
    if 'en_vectors' not in st.session_state:
        print('Initial loading..')
        print('Model:', add_selectbox)
        
        doc = pickle.load(open('../../data/word2vec/europarl/europarl_df.pkl', 'rb'))
        model_path = '../../data/word2vec/europarl/models/contextual/make-multilingual-sys-2021-07-24_21-03-57'
        europarl_corpus = corpus.Corpus(name='europarl', 
                                        doc=doc)        
        st.session_state['doc_df'] = doc
        st.session_state['corpus'] = europarl_corpus
        
        if add_selectbox == 'Static Embeddings':       
            st.session_state['w2v'] = word2vec.Word2Vec(name='europarl',
                                    corpus=europarl_corpus,
                                    rerun_data=False,
                                    rerun_train=False
                                   )
            
            st.session_state['model'] = SentenceTransformer(model_path)
            
            st.session_state['static_ret'] = retrieval.Retrieval(corpus=europarl_corpus,
                                                                 model=st.session_state['w2v'],
                                                                 model_type='static')
            
            st.session_state['cont_ret'] = retrieval.Retrieval(corpus=europarl_corpus,
                                                                model=st.session_state['model'],
                                                                model_type='contextual')
            
            
            # docs = static_ret.get_retrieved_docs(query='presidential elections', num_ret=5)
            # print(docs)
            # 
            # docs = cont_ret.get_retrieved_docs(query='presidential elections', num_ret=5)
            # print(docs)            
            
        # en_vectors = pickle.load(open(EN_VECTORS_PATH, 'rb'))
        # de_vectors = pickle.load(open(DE_VECTORS_PATH, 'rb'))
        proj_2d = pickle.load(open(PROJ_PATH, 'rb'))
        st.session_state['proj_2d'] = proj_2d

        stop_words = set(stopwords.words('english'))
        stop_words_de = set(stopwords.words('german'))
        punct = list(string.punctuation)
        lemmatizer_en = WordNetLemmatizer()
        lemmatizer_de = Cistem()  

        st.session_state['stop_words'] = stop_words
        st.session_state['stop_words_de'] = stop_words_de
        st.session_state['punct'] = punct
        st.session_state['lemmatizer_en'] = lemmatizer_en
        st.session_state['lemmatizer_de'] = lemmatizer_de
        
        print('Initializing clusters')
        doc_df = st.session_state['doc_df']
        corpus_en = st.session_state['corpus'].corpus_en
        corpus_de = st.session_state['corpus'].corpus_de        
        st.session_state['fig_2d'], st.session_state['df'], st.session_state['colors'] = get_clusters()

        
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Embedding model',
    ('Static Embeddings', 'Contextual Embeddings')
)

print('Model:', add_selectbox)


# Add a slider to the sidebar:
NUM_RETRIEVAL = st.sidebar.slider(
    'Select number of documents to be retrieved',
    3, 10, 5
)

st.write("""
# Explainable Cross-lingual Text Retrieval on Automotive domain
# """)

load_model()

print('Reloading..')
doc_df = st.session_state['doc_df']
corpus = st.session_state['corpus'].corpus_en
corpus_de = st.session_state['corpus'].corpus_de

stop_words = st.session_state['stop_words']
stop_words_de = st.session_state['stop_words_de']
punct = st.session_state['punct']
lemmatizer_en = st.session_state['lemmatizer_en']
lemmatizer_de = st.session_state['lemmatizer_de']
    
if add_selectbox == 'Static Embeddings':
    en_vectors = st.session_state['static_ret'].en_doc_vectors
    de_vectors = st.session_state['static_ret'].de_doc_vectors
    proj_2d = st.session_state['proj_2d']
    model = st.session_state['w2v']
    ret = st.session_state['static_ret']
else:
    en_vectors = st.session_state['static_ret'].en_doc_vectors
    de_vectors = st.session_state['static_ret'].de_doc_vectors
    proj_2d = st.session_state['proj_2d']
    model = st.session_state['w2v']
    ret = st.session_state['cont_ret']

print('Reloading clusters..')
fig_2d = st.session_state['fig_2d']
df = st.session_state['df']
colors = st.session_state['colors']

st.write('### Enter the query')
query_input = st.text_area(label='Enter the query and type Ctrl+Enter to search')
# if (st.button('Search')):
now = datetime.datetime.now()
out = ret.get_retrieved_docs(query_input, NUM_RETRIEVAL)
print('out', out)
after = datetime.datetime.now()
difference = after - now

st.write('### Search results for "%s" in %f microseconds'%(query_input, difference.microseconds))
docs_en = out['en_docs']
docs_de = out['de_docs']
docs = docs_en + docs_de

# for i in y_pred_en:
#     docs.append(corpus[i])
#     docs_en.append(corpus[i])
# 
# for i in y_pred_de:
#     docs.append(corpus_de[i])
#    docs_de.append(corpus_de[i])

with st.form(key='my_form'):
    doc_selected = st.radio("", tuple(docs))
    submit_button = st.form_submit_button(label='Explain')

def get_sim_with_replacement(first, second, second_lang):   
    second_tokens = tokenize.get_tokens(second, lang=second_lang)
    
    sentences = [first, second]
    for token in second_tokens:
        repl = re.sub(token, ' ', second)
        repl = re.sub(token.capitalize(), ' ', repl)
        sentences.append(repl)
    
    print('here', type(sentences))
    vectors = ret.get_sentence_vector(sentences)
    
    # output[0] --> similarity with selected doc
    # output[1:] --> similarity with token replacement
    output = F.cosine_similarity(vectors[0].unsqueeze(0), vectors[1:], dim=1)
    orig_sim = output[0]    
    diff_in_sim = torch.sub(orig_sim, output[1:])
    diff_in_sim = diff_in_sim.cpu().numpy()
    print('diff_in_sim', diff_in_sim)
    
    tokens_import = {token:diff for token, diff in zip(second_tokens, diff_in_sim)}
    print(tokens_import)
    
    return tokens_import


def plot_import_bar(tokens_import):
    df = []
    for token in tokens_import:
        importance = tokens_import[token]
        df.append(
            {
                'token': token,
                'importance': importance
            }
        )
    df = pd.DataFrame(df)
    d = {}
    for val in df['importance'].values:
        if val >= 0:
            d[val] = 'green'    
        else:
            d[val] = 'red' 
    colors = [d[k] for k in df['importance'].values]    
    
    fig = px.bar(df, x="importance", y="token", orientation='h', height=500)
    fig.update_traces(marker_color=colors) 
    fig.add_vrect(x0=0, x1=0)
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    return fig

def update_clusters(docs, doc_selected):
    fig_2d = st.session_state['fig_2d']
    df = st.session_state['df']
    colors = st.session_state['colors']
    
    colors_new = colors[:]
    for doc in docs:
        ind = list(df['text'].values).index(doc)
        colors_new[ind] = 'red'
        
    ind = list(df['text'].values).index(doc_selected)
    colors_new[ind] = 'green'

    fig_2d.update_traces(hoverlabel=dict(align="left"), marker_color=colors_new)
    return fig_2d
    

def update_token_importance():
    if doc_selected in docs_en:
        lang = 'en'
    else:
        lang = 'de'
    
    tokens_import = get_sim_with_replacement(query_input, doc_selected, second_lang=lang)
    plt = plot_import_bar(tokens_import)
    
    return plt

    
with st.spinner('Please wait while we fetch the explanations..'):
    fig_2d = update_clusters(docs, doc_selected)
    plt = update_token_importance()
    st.success('Loaded explanations')    
    
st.write("### Explaining retrieval of")
st.write("`%s`"%doc_selected)

with st.expander("Cluster analysis"):
    st.plotly_chart(fig_2d, use_container_width=True)

with st.expander("Word contribution"):
    st.plotly_chart(plt, use_container_width=True)
