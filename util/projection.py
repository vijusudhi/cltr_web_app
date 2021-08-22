from tqdm.notebook import tqdm as tq

def generate_tsv_files(w2v):
    vec_file = open('vectors_%s.tsv'%(w2v.name), 'w')
    words_file = open('words_%s.tsv'%(w2v.name), 'w')
    for word in tq(w2v.wv_vocab):
        # vec = w2v.model.wv[word]
        vec = w2v.model.syn1neg[w2v.wv_vocab.index(word)] + w2v.model.wv[word]
        vec_str = ''
        for i in vec:
            vec_str += str(i) + '\t'
        vec_file.write(vec_str.strip()+ '\n')
        words_file.write(word.strip()+ '\n')
    vec_file.close()
    words_file.close()
    print('Vectors saved at: %s' %('vectors_%s.tsv'%(w2v.name)))
    print('Words saved at: %s' %('words_%s.tsv'%(w2v.name)))