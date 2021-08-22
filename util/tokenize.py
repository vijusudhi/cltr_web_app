import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.cistem import Cistem

# define lemmatizer for EN and DE
lemmatizer_en = WordNetLemmatizer()
lemmatizer_de = Cistem()

# define stop words for EN and DE
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))

# define punctuations
punct = list(string.punctuation)
punct.remove('-')
punct.remove('_')

def get_tokens(text, lang='en', lemmatize=False):
    '''
    This function tokenizes and returns the unique tokens
    '''
    tokens = nltk.word_tokenize(text.lower())
    
    stop_words = stop_words_en if lang=='en' else stop_words_de
    lemmatizer = lemmatizer_en.lemmatize if lang=='en' else lemmatizer_de.stem
    
    pre_proc_tokens = []
    for token in tokens:
        punct_found = False
        if token.isdigit() or token in stop_words or token in punct:
            continue
        for p in punct:
            if p in token:
                punct_found = True
                break
        if punct_found:
            continue
        
        if lemmatize:
            lemma = lemmatizer(token)
            if lemma in pre_proc_tokens:
                continue
            pre_proc_tokens.append(lemma)
        else:
            if token in pre_proc_tokens:
                continue
            pre_proc_tokens.append(token)
            
    return pre_proc_tokens