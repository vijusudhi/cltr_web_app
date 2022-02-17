import pandas as pd
from os import walk
import numpy as np
import re
import sys
sys.path.append('../')
from util import util
import pprint
pp = pprint.PrettyPrinter(indent=4)

class EvaluateExplanations:
    def __init__(self, path) -> None:
        self.path = path
        self.filenames = self.get_log_files()
        dfs = [self.get_df(filename) for filename in self.filenames]
        valid = [self.check_if_valid_log(df) for df in dfs]
        self.dfs = self.filter_dfs(dfs, valid)

        self.sys_gen_words = util.decompress_pickle('sys_gen_words')
        self.all_words = util.decompress_pickle('all_words')


    @staticmethod
    def get_df(filename):
        df = pd.read_csv(filename)
        df.columns = ['q_id', 'query_text', 'is_relevant', 'relevant_words']
        return df   

    @staticmethod
    def check_if_valid_log(df):
        if len(df['q_id'].to_list()) > 5:
            return True
        else:
            return False

    @staticmethod
    def check_if_valid_log(df):
        if len(df['q_id'].to_list()) > 5:
            return True
        else:
            return False

    @staticmethod
    def filter_dfs(dfs, valid):
        return [df for ind, df in enumerate(dfs) if valid[ind]]
    
    def get_log_files(self):
        """
        checks the logs path and returns only the csv files
        """
        _, _, filenames = next(walk(self.path))
        print(filenames)
        filenames = [f"{self.path}/{filename}" for filename in filenames if "csv" in filename]
        return filenames

    
    def get_accuracy(self, df):
        """
        calculate number of Yes and No in is_relevant.
        This helps to understand how the user thinks results from
        static embeddings and contextual embeddings
        """
        st_yes, st_no = 0, 0
        ct_yes, ct_no = 0, 0
        for _, row in df.iterrows():
            if 'ST' in row['q_id']:
                if row['is_relevant'].strip() == 'Yes':
                    st_yes += 1
                    self.st_queries_counts[row['q_id'].strip()][1] += 1
                else:
                    st_no += 1
                    self.st_queries_counts[row['q_id'].strip()][0] += 1
                    
            if 'CT' in row['q_id']:
                if row['is_relevant'].strip() == 'Yes':
                    ct_yes += 1
                    self.ct_queries_counts[row['q_id'].strip()][1] += 1
                else:
                    ct_no += 1
                    self.ct_queries_counts[row['q_id'].strip()][0] += 1
                    
        metrics = {
            "st_yes": st_yes,
            "st_no": st_no,
            "ct_yes": ct_yes,
            "ct_no": ct_no
        }
        return metrics

    def get_total_counts(self):
        count = lambda metric : np.sum(np.asarray([a[metric] for a in self.acc]))
        st_yes = count('st_yes')
        st_no = count('st_no')
        ct_yes = count('ct_yes')
        ct_no = count('ct_no')
        self.total_counts = {
            "st_yes": st_yes,
            "st_no": st_no,
            "ct_yes": ct_yes,
            "ct_no": ct_no
        }
    
    def init_per_query_counts(self):
        """
        0 --> no, 1 --> yes
        """
        self.st_queries_counts = {"ST_{:>03}".format(id):[0,0] for id in range(1, 31)}
        self.ct_queries_counts = {"CT_{:>03}".format(id):[0,0] for id in range(1, 31)} 

    def report_majority_counts(self):
        st_yes, st_no, ct_yes, ct_no = 0, 0, 0, 0
        for id in range(1, 31):
            st = self.st_queries_counts["ST_{:>03}".format(id)]
            ct = self.ct_queries_counts["CT_{:>03}".format(id)]
            if st[0] > st[1]:
                st_no += 1
            else:
                st_yes += 1

            if ct[0] > ct[1]:
                ct_no += 1
            else:
                ct_yes += 1

        self.majority_counts = {
            "st_yes": st_yes,
            "st_no": st_no,
            "ct_yes": ct_yes,
            "ct_no": ct_no
        } 

    @staticmethod
    def get_accuracy_values(counts):
        acc = {
            "st_acc": counts['st_yes'] / (counts['st_yes'] + counts['st_no']),
            "ct_acc": counts['ct_yes'] / (counts['ct_yes'] + counts['ct_no'])
        }
        return acc

    @staticmethod
    def clean_user_relevant_words(relevant_words):
        words = relevant_words.split(":")
        words = [re.sub("\(.*\)", "", word) for word in words]
        words = [word.strip() for word in words]
        return words

    def get_metrics_for_all_queries(self, df):
        prec_st, recall_st, f_score_st = 0, 0, 0
        prec_ct, recall_ct, f_score_ct = 0, 0, 0
        count = 0
        for _, row in df.iterrows():
            q_id = row['q_id'].strip()
            query_text = row['query_text'].strip()
            is_relevant = row['is_relevant'].strip()
            relevant_words = row['relevant_words'].strip()

            if is_relevant == 'Yes':
                count += 1
                words = relevant_words.split(":")
                words = [re.sub("\(.*\)", "", word) for word in words]
                words = [word.strip() for word in words]
                metrics = self.get_metrics_per_query(user_words=words,
                                                sys_words=self.sys_gen_words[q_id],
                                                all_words=self.all_words[q_id])
                
                if "ST_" in q_id:
                    prec_st += metrics['prec']
                    recall_st += metrics['recall']
                    f_score_st += metrics['f_score']
                else:
                    prec_ct += metrics['prec']
                    recall_ct += metrics['recall']
                    f_score_ct += metrics['f_score']                
                
        try:
            macro_prec_st = prec_st/count
            macro_recall_st = recall_st/count
            macro_f_score_st = f_score_st/count
            
            macro_prec_ct = prec_ct/count
            macro_recall_ct = recall_ct/count
            macro_f_score_ct = f_score_ct/count        
        except:
            macro_prec_st, macro_recall_st, macro_f_score_st = 0, 0, 0
            macro_prec_ct, macro_recall_ct, macro_f_score_ct = 0, 0, 0
            # print("Count is zero. No row marked relevant. Cross-check!")
        
        metrics = {
            "static":
            {
                "prec": macro_prec_st, 
                "recall": macro_recall_st, 
                "f_score": macro_f_score_st
            },
            "contextual":
            {
                "prec": macro_prec_ct, 
                "recall": macro_recall_ct, 
                "f_score": macro_f_score_ct
            }

        }
        return metrics

    @staticmethod
    def get_metrics_per_query(user_words, sys_words, all_words):
        tp, tn, fp, fn = 0, 0, 0, 0
        for word in all_words:
            if word in user_words and word in sys_words:
                tp += 1
            if word not in user_words and word not in sys_words:
                tn += 1
            if word in user_words and word not in sys_words:
                fn += 1
            if word not in user_words and word in sys_words:
                fp += 1
        
        prec, recall, f_score = 0, 0, 0
        if tp+fp:
            prec = tp/(tp+fp)
        if tp+fn:
            recall = tp/(tp+fn)
        if prec+recall:
            f_score = 2*((prec*recall)/(prec+recall))

        metrics = {
            "prec": prec,
            "recall": recall,
            "f_score": f_score
        }
        return metrics

    def get_macro_averaged_metrics(self):
        macro_prec_st, macro_recall_st, macro_f_score_st = 0, 0, 0
        macro_prec_ct, macro_recall_ct, macro_f_score_ct = 0, 0, 0

        for df in self.dfs:
            metrics = self.get_metrics_for_all_queries(df)
            macro_prec_st += metrics['static']['prec']
            macro_recall_st += metrics['static']['recall']
            macro_f_score_st += metrics['static']['f_score']

            macro_prec_ct += metrics['contextual']['prec']
            macro_recall_ct += metrics['contextual']['recall']
            macro_f_score_ct += metrics['contextual']['f_score']            

        avg = lambda metric: metric/len(self.dfs)
        macro_prec_st = avg(macro_prec_st)
        macro_recall_st = avg(macro_recall_st)
        macro_f_score_st = avg(macro_f_score_st)

        macro_prec_ct = avg(macro_prec_ct)
        macro_recall_ct = avg(macro_recall_ct)
        macro_f_score_ct = avg(macro_f_score_ct)        

        metrics = {
            "static":
            {
                "prec": macro_prec_st, 
                "recall": macro_recall_st, 
                "f_score": macro_f_score_st
            },
            "contextual":
            {
                "prec": macro_prec_ct, 
                "recall": macro_recall_ct, 
                "f_score": macro_f_score_ct
            }
        }

        return metrics
        

if __name__ == '__main__':
    eval = EvaluateExplanations(path=r"D:\master_thesis_viju\cltr_web_app\logs")

    # EVAL 01 - Static or Contextual? Which gives better results?
    eval.init_per_query_counts()
    eval.acc = [eval.get_accuracy(df) for df in eval.dfs]
    # total counts just take the sum total of Yes and No responses
    eval.get_total_counts()
    # since user reponses vary significantly, it is better to consider the
    # majority of response for each query.
    # ST_IDX --> Yes if majority of users marked Yes, else No
    # Similarly for CT_IDX
    # This is calculated with majority counts
    eval.report_majority_counts()
    print("Total counts")
    pp.pprint(eval.get_accuracy_values(eval.total_counts))
    print("Majority counts")
    pp.pprint(eval.get_accuracy_values(eval.majority_counts))    
    
    # EVAL 02 - Static or Contextual? Which gives better explanations?
    print("Macro-averaged metrics")
    pp.pprint(eval.get_macro_averaged_metrics())