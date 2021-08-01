import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.cistem import Cistem
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm as tq
import string
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import datetime
import umap.umap_ as umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import pandas as pd
import textwrap
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd

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
    return sent_vector.unsqueeze(0)
    
def get_retrieved_docs(query):
    # Compute the target vectors before-hand. These do not change.   
    query_vector = get_sentence_vector(model, query)
    output = F.cosine_similarity(query_vector, de_vectors, dim=1)
    y_pred_de = torch.topk(output, NUM_RETRIEVAL).indices.cpu().numpy()
    
    output = F.cosine_similarity(query_vector, en_vectors, dim=1)
    y_pred_en = torch.topk(output, NUM_RETRIEVAL).indices.cpu().numpy()    
    return y_pred_en, y_pred_de

@st.cache
def load_model():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    def download_file(file_path, drive_link, unzip=False):
        file = Path(file_path)
        if not file.exists():
            with st.spinner("Downloading %s. Please wait..." % file_path):
                gdd.download_file_from_google_drive(file_id=drive_link,
                                        dest_path=file_path,
                                        unzip=unzip)        
    
    download_file(file_path=PROJ_PATH, drive_link=PROJ_LINK, unzip=False)
    download_file(file_path=EN_VECTORS_PATH, drive_link=EN_VECTORS_LINK, unzip=False)
    download_file(file_path=DE_VECTORS_PATH, drive_link=DE_VECTORS_LINK, unzip=False)
    download_file(file_path=EUROPARL_DATA_PATH, drive_link=EUROPARL_DATA_LINK, unzip=False)
    download_file(file_path=CONT_MODEL_PATH, drive_link=CONT_MODEL_LINK, unzip=True)    

load_model()

en_vectors = pickle.load(open(EN_VECTORS_PATH, 'rb'))
de_vectors = pickle.load(open(DE_VECTORS_PATH, 'rb'))
proj_2d = pickle.load(open(PROJ_PATH, 'rb'))    

doc_df = pickle.load(open(EUROPARL_DATA_PATH, 'rb'))
corpus = doc_df.en.to_list()
corpus_de = doc_df.de.to_list()

model = SentenceTransformer(CONT_MODEL_PATH)

stop_words = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))
punct = list(string.punctuation)
lemmatizer_en = WordNetLemmatizer()
lemmatizer_de = Cistem()


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Embedding model',
    ('Static Embeddings', 'Contextual Embeddings')
)

# Add a slider to the sidebar:
NUM_RETRIEVAL = st.sidebar.slider(
    'Select number of documents to be retrieved',
    3, 10, 5
)

st.write("""
# Explainable Cross-lingual Text Retrieval on Automotive domain
# """)

st.write('### Enter the query')
query_input = st.text_area(label='Enter the query and type Ctrl+Enter to search')
# if (st.button('Search')):
now = datetime.datetime.now()
y_pred_en, y_pred_de = get_retrieved_docs(query_input)
after = datetime.datetime.now()
difference = after - now

st.write('### Search results for "%s" in %f seconds'%(query_input, difference.seconds))
docs = []
docs_en = []
docs_de = []
for ind, i in enumerate(y_pred_en):
    docs.append(corpus[i])
    docs_en.append(corpus[i])

for ind, i in enumerate(y_pred_de):
    docs.append(corpus_de[i])
    docs_de.append(corpus_de[i])

with st.form(key='my_form'):
    doc_selected = st.radio("", tuple(docs))
    submit_button = st.form_submit_button(label='Explain')

def get_sim_with_replacement(first, second, second_lang):
    first_vector = get_sentence_vector(model, first)
    second_vector = get_sentence_vector(model, second)
    output = F.cosine_similarity(first_vector, second_vector, dim=1)
    orig_sim = output[0]
    
    second_tokens = get_tokens_nltk(second, lang=second_lang)
    tokens_import = {}
    for token in second_tokens:
        repl = re.sub(token, ' ', second)
        repl = re.sub(token.capitalize(), ' ', second)
        vec = get_sentence_vector(model, repl)
        output = F.cosine_similarity(first_vector, vec, dim=1)
        diff_in_sim = orig_sim - output[0]
        tokens_import[token] = diff_in_sim
    
    return tokens_import

def get_tokens_nltk(text, lang='en', lemmatize=False):
    '''
    this function tokenizes and returns the unique tokens
    '''
    tokens = nltk.word_tokenize(text.lower())
    
    if lang == 'en':
        stop_words_xx = stop_words
    elif lang == 'de':
        stop_words_xx = stop_words_de
    
    pre_proc_tokens = []
    for token in tokens:
        punct_found = False
        if token.isdigit() or token in stop_words_xx or token in punct:
            continue
        for p in punct:
            if p in token:
                punct_found = True
                break
        if punct_found:
            continue
        
        if lemmatize:
            if lang == 'en': 
                lemma = lemmatizer_en.lemmatize(token)
            else:
                lemma = lemmatizer_de.stem(token)
            if lemma in pre_proc_tokens:
                continue
            pre_proc_tokens.append(lemma)
        else:
            if token in pre_proc_tokens:
                continue
            pre_proc_tokens.append(token)
            
    return pre_proc_tokens


def plot_import_bar(tokens_import):
    # fig = plt.figure(figsize = (20, 10))
    # 
    # fig = px.bar(df, orientation='h')
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

# @st.cache(show_spinner=False)
def get_explanations():
    # Plot UMAP clusters
    df = pd.DataFrame(corpus[:5000]+corpus_de[:5000])
    df.columns = ['text']

    d = {}
    for val in df['text'].values:
        if val == doc_selected:
            d[val] = 'green'    
        elif val in docs:
            d[val] = 'red'
        else:
            d[val] = 'lavender' 
    colors = [d[k] for k in df['text'].values]
    
    # Uncomment if to run on fly
    # vectors = []
    # vectors.extend(en_vectors)
    # vectors.extend(de_vectors)
    # vectors = torch.stack(vectors)
    
    # umap_2d = UMAP(n_components=2, init='random', random_state=0)
    # proj_2d = umap_2d.fit_transform(vectors)

    fig_2d = px.scatter(
        proj_2d, x=0, y=1,
        hover_name=df['text'].apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=50))))


    fig_2d.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
    fig_2d.update_traces(hoverlabel=dict(align="left"), marker_color=colors)    

    if doc_selected in docs_en:
        lang = 'en'
    else:
        lang = 'de'

    tokens_import = get_sim_with_replacement(query_input, doc_selected, second_lang=lang)
    plt = plot_import_bar(tokens_import)
    
    return fig_2d, plt


# with st.spinner('Please wait while we fetch the explanations..'):
#    time.sleep(10)
    
with st.spinner('Please wait while we fetch the explanations..'):
    fig_2d, plt = get_explanations()
    st.success('Loaded explanations')    
    
st.write("### Explaining retrieval of")
st.write("`%s`"%doc_selected)

with st.beta_expander("Cluster analysis"):
    st.plotly_chart(fig_2d, use_container_width=True)

with st.beta_expander("Word contribution"):
    st.plotly_chart(plt, use_container_width=True)
