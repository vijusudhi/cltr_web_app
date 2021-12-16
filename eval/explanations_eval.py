import pandas as pd
from os import walk
import numpy as np

class EvaluateExplanations:
    def __init__(self, path) -> None:
        self.path = path
        self.filenames = self.get_log_files()
        dfs = [self.get_df(filename) for filename in self.filenames]
        valid = [self.check_if_valid_log(df) for df in dfs]
        self.dfs = self.filter_dfs(dfs, valid)

        self.init_per_query_counts()
        self.acc = [self.get_accuracy(df) for df in self.dfs]
        # total counts just take the sum total of Yes and No responses
        self.get_total_counts()
        # since user reponses vary significantly, it is better to consider the
        # majority of response for each query.
        # ST_IDX --> Yes if majority of users marked Yes, else No
        # Similarly for CT_IDX
        # This is calculated with majority counts
        self.report_majority_counts()

        print("Total counts", self.get_accuracy_values(self.total_counts))
        print("Majority counts", self.get_accuracy_values(self.majority_counts))


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
                ct_yes += 1
            else:
                ct_no += 1

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

if __name__ == '__main__':
    eval = EvaluateExplanations(path="/home/sudhi/thesis/cltr_web_app/logs")
    