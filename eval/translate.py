import pandas as pd
import sys
sys.path.append('../')
from cltr_web_app.util import tokenize
import re

group = "b"
inp = pd.read_csv('eval/group_%s_eval.csv' % group, header=None, 
              names=['q_id', 'query_text', 'document', 'lang'],
              delimiter=',')

# for _, row in inp.iterrows():
#     print(row['query_text'])
# 
# for _, row in inp.iterrows():
#     print(row['document'])
#     print("---")

file = open("eval/group_%s_translation.txt" % group, 'r', encoding='utf-8')
lines = file.readlines()
lines = [line.strip() for line in lines]
file.close()

for ind, line in enumerate(lines):
    if line == "QUERIES":
        queries_start = ind
    if line == "DOCUMENTS":
        docs_start = ind
    if line == "KEYWORDS":
        keywords_start = ind

print(queries_start, lines[queries_start])
print(docs_start, lines[docs_start])
print(keywords_start, lines[keywords_start])

queries_translated = [line for line in lines[queries_start+1:docs_start] if line != ""]
docs_translated = [line for line in lines[docs_start+1:keywords_start] if line != ""]
keywords_translated = [line for line in lines[keywords_start+1:len(lines)] if line != ""]

keywords_translated = []
for line in lines[keywords_start+1:len(lines)]:
    line = line.strip()
    keywords = line.split(",")
    keywords_ref = []
    for word in keywords:
        word = re.sub("\'", "", word)
        word = re.sub("\[", "", word)
        word = re.sub("\]", "", word)
        word = word.strip()
        keywords_ref.append(word)
    keywords_translated.append(keywords_ref)


assert len(queries_translated) == len(docs_translated), "Length of translations not right"
assert len(keywords_translated) == len(docs_translated), "Length of translations not right"

eval_extended = []
for idx, row in inp.iterrows():
    eval_extended.append(
        {
            'q_id': row['q_id'],
            'query_text': row['query_text'],
            'query_text_tr': queries_translated[idx],
            'document': row['document'],
            'document_tr': docs_translated[idx],
            'lang': row['lang'],
            'keywords_tr': keywords_translated[idx]
        }
    )

eval_extended_df = pd.DataFrame(eval_extended)
eval_extended_df.to_csv("eval/group_%s_extended.csv" % group, index=False)

# inp = pd.read_csv('eval/group_%s_extended.csv' % group)
# for _, row in inp.iterrows():
#     words = tokenize.get_tokens(row['document'], lang=row['lang'])
#     print(words)