import streamlit as st
import datetime
import re
from util import util, explain_cont, states

NUM_RETRIEVAL = 5

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("css/style.css") 

@st.cache(allow_output_mutation=True)
def load_states():
    scraped_df = util.decompress_pickle('data/scraped_df')

    cached_state = states.CachedState(
                                scraped_df = scraped_df,
                            )
    
    return cached_state


def display_header():
    col1, _, col2 = st.columns([3,1,20])
    with col1:
        st.image('data/multilingual-icon-9.jpg', width=100)
    with col2:
        st.write("""
        # Explainable Cross-lingual Text Retrieval in the Automotive domain
        # """)        


def page_home():
    if 'app_state' in st.session_state:
        app_state = st.session_state['app_state']
        
        query_text = app_state.query
        query = app_state.query
        query_lang = app_state.query_lang
        query_lang_corrected = app_state.query_lang_corrected
        retrieved = app_state.retrieved
        token_importance_en = app_state.token_importance_en
        token_importance_de = app_state.token_importance_de
    else:
        query_text = ''
        query_lang_corrected = False
        

    with st.spinner('Please wait while we load the environment..'):
        cached_state = load_states()
        
    st.write('### Enter the query')       
    
    query = st.text_input(label='Enter the query and type Ctrl+Enter to search',
                         value=query_text)
    
    if query != '':
        if query_text != query:
            query_lang_corrected = False

        query_us = re.sub(' ', '_', query) 
        query_lang = 'en'

        with st.container():
            col1, col2 = st.columns([5, 5])
            with col1:
                st.markdown(f"Query language identified as: <span class='highlight red_bold'>{query_lang}</span>", unsafe_allow_html=True)    

        now = datetime.datetime.now()   

        suggestions = util.decompress_pickle(f"dump/{query_us}_suggestions")
        st.write("### You may also search for")
        with st.container():
            text = ""
            length = 0
            for suggestion in suggestions:
                length += len(suggestion)
                # check if the length of string exceeds container size
                # add just the span if it does not
                # else, write the text and reinitialize the values
                if length <= 75:
                    text += f"<span class='highlight red'>{suggestion}</span>"
                else:
                    st.markdown(text, unsafe_allow_html=True)
                    text = ""
                    length = len(suggestion)
                    text += f"<span class='highlight red'>{suggestion}</span>"
            # write any pieces of text remaining
            st.markdown(text, unsafe_allow_html=True)

        after = datetime.datetime.now()
        difference = after - now            

        retrieved = util.decompress_pickle(f"dump/{query_us}_retrieved")

        st.write('### Search results for "%s" in %f microseconds'%(query, difference.microseconds))

        token_importance_en = util.decompress_pickle(f"dump/{query_us}_token_importance_en")         
        token_importance_de = util.decompress_pickle(f"dump/{query_us}_token_importance_de")                       

        app_state = states.AppState(
                                query = query,
                                query_lang = query_lang,
                                query_lang_corrected = query_lang_corrected,
                                retrieved = retrieved,
                                token_importance_en = token_importance_en,
                                token_importance_de = token_importance_de,
                            )

        st.session_state['app_state'] = app_state

        with st.expander("Explore representation space"):
            repr_space = util.decompress_pickle(f"dump/{query_us}_repr_space")
            imp_word = util.decompress_pickle(f"dump/{query_us}_imp_word")
            text = f'\
            You can see the representation space of the queries and documents below.<br>\
            <span style="color: transparent;  text-shadow: 0 0 0 green; ">&#9899;</span> Query\
            <span style="color: transparent;  text-shadow: 0 0 0 red; ">&#9899;</span> Document <b>relevant</b> to the query <br>\
            <span style="color: transparent;  text-shadow: 0 0 0 blue; ">&#9899;</span> Document with word \
            <span class="highlight red">{imp_word}</span><b>not relevant</b> to the query<br>\
            Size of the markers indicate contextual similarity.\
            '
            st.markdown(text, unsafe_allow_html=True)
            if repr_space == -1:
                st.markdown('Sorry! Can not dsiplay the space', unsafe_allow_html=True)
            else:                   
                st.plotly_chart(repr_space, use_container_width=True)

        display_search_results(cached_state, doc_lang='en')
        display_search_results(cached_state, doc_lang='de')

    
def display_search_results(cached_state, doc_lang):
    if doc_lang == 'en':
        sim = st.session_state['app_state'].retrieved['en_sim']
        docs = st.session_state['app_state'].retrieved['en_docs']
        token_imp = st.session_state['app_state'].token_importance_en
        key_ind = 0
    else:
        sim = st.session_state['app_state'].retrieved['de_sim']
        docs = st.session_state['app_state'].retrieved['de_docs']
        token_imp = st.session_state['app_state'].token_importance_de
        key_ind = 100
    
    for sim, doc in zip(sim, docs):
        html_string = explain_cont.get_display_text(doc, token_imp, mode='bold')
        url, title = explain_cont.get_url(cached_state.scraped_df, doc)
        with st.container():
            col1, mid = st.columns([2, 20])
            explain_state = states.ExplainState(
                                doc=doc,
                                doc_lang=doc_lang,
                                sim=sim
                            )
            with col1:
                st.button(label="X", key='%d'%key_ind, 
                                   on_click=update_and_explain,
                                   args=(explain_state,)
                                  )
                key_ind += 1
            with mid:
                st.markdown('[%s](%s)'%(title, url))    
                st.markdown(html_string, unsafe_allow_html=True)
    
def update_and_explain(explain_state):
    st.session_state["page"] = 'Explanations'
    st.session_state['explain_state'] = explain_state
    
def page_explanations():    
    query = st.session_state['app_state'].query    
    doc = st.session_state['explain_state'].doc
    doc_lang = st.session_state['explain_state'].doc_lang
    
    st.write("### Explaining retrieval of:")
    with st.container():
        col1, col2 = st.columns([5, 20])
        col1.markdown("Query")
        col2.markdown(f"<span class='highlight red_bold'>{query}</span>", unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([5, 20])
        col1.markdown("Document")
        col2.markdown(f"**{doc}**", unsafe_allow_html=True)
    
    query_us = re.sub(' ', '_', query)
    retrieved = util.decompress_pickle(f"dump/{query_us}_retrieved")

    if doc in retrieved['en_docs']:
        doc_idx = retrieved['en_docs'].index(doc)
    if doc in retrieved['de_docs']:
        doc_idx = retrieved['de_docs'].index(doc)

    # st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.expander('Explanation 02 - Query-Document terms co-occurrences'):
        with st.container():
            col1, col2, col3 = st.columns([10, 1, 30])
            col1.markdown("<span class='heading'><b>Query-Document terms co-occurrences</b></span>", unsafe_allow_html=True)
            col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
            col3.markdown("<p>The model was trained on patents from the European Patent Office (EPO) belonging to the International Patent Classification (IPC) <i>B60 Vehicles in General</i>.</p>",
                        unsafe_allow_html=True
                        )                   
            col3.markdown("<p>You can see the query-document terms co-occurrences found the corpus below.</p>",
                        unsafe_allow_html=True
                        ) 

        heatmap = util.decompress_pickle(f"dump/{query_us}_{doc_lang}_{doc_idx}_heatmap")
        st.plotly_chart(heatmap, use_container_width=True)        

    # st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.expander('Explanation 03 - Query-Document term associations'):
        with st.container():
            col1, col2, col3 = st.columns([10, 1, 30])
            col1.markdown("<span class='heading'><b>Query-Document term associations</b></span>", unsafe_allow_html=True)
            col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
            col3.markdown("<p>The model knows both English and German <i>reasonably well</i>. It can say which pair of words associate with one another.</p>",
                        unsafe_allow_html=True
                        )
            col3.markdown("<p>You can see below <span class='highlight darkbrown_bold'>high</span> to <span class='highlight lightbrown_bold'>low</span> associations of document terms with the query terms.</p>",
                    unsafe_allow_html=True
                    )    

        spit_imp = util.decompress_pickle(f"dump/{query_us}_{doc_lang}_{doc_idx}_spit_imp")   
        st.markdown("<p></p>", unsafe_allow_html=True)   
        with st.container():
            col_q, col_txt = st.columns([5, 20])
            col_q.markdown('*Query term*')
            col_txt.markdown('*Document*')
        for i in spit_imp:
            with st.container():
                # TODO: update column width dynamically according to size
                # of text
                col_q, col_txt = st.columns([5, 20])
                with col_q:
                    st.markdown('**%s**' %i['split'])
                with col_txt:
                    st.markdown(i['text'], unsafe_allow_html=True)        

    # st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    with st.expander('Explanation 04 - Document term significance'):
        with st.container():
            col1, col2, col3 = st.columns([10, 1, 30])
            col1.markdown("<span class='heading'><b>Document term significance</b></span>", unsafe_allow_html=True)
            col2.markdown("<div class= 'vertical'></div>", unsafe_allow_html=True)
            col3.markdown("<p>Each document term contribute differently to the retrieval of this document. It can either prompt the system to retrieve the document or otherwise.</p>",
                        unsafe_allow_html=True
                        )
            col3.markdown("<p> You can see below the <span class='highlight darkgreen_bold'>positive</span> or <span class='highlight darkred_bold'>negative</span> contribution of document terms to the retrieval.</p>",
                    unsafe_allow_html=True
                    )                     

        
        plt_imp = util.decompress_pickle(f"dump/{query_us}_{doc_lang}_{doc_idx}_plt_imp")
        st.plotly_chart(plt_imp, use_container_width=True)                    
    
    st.session_state["page"] = 'Home'
    st.button(label="Exit", key='626', on_click=update_and_exit)

    
def update_and_exit():
    st.session_state["page"] = 'Home'

PAGES = {
    "Home": page_home,
    "Explanations": page_explanations,
}           
    
def main():    
    if "page" not in st.session_state:
        st.session_state.update(
            {
                'page': 'Home'
            }
        )
        
    PAGES[st.session_state['page']]()


DEBUG_MODE = True
if __name__ == "__main__":
    if DEBUG_MODE:
        display_header()
        main()
    else:
        try:
            display_header()
            main()
        except:
            st.error('Oh snap! Error!')