import pandas as pd
from gensim.summarization import summarize
import warnings
warnings.filterwarnings('ignore')

class JobSummary:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def sentence_count_gensim(self, text):
        from gensim.summarization.textcleaner import split_sentences
        return len(split_sentences(text))

    def process_data(self):
        df = pd.read_csv(self.input_file)
        counts = df['job'].value_counts()
        under10 = counts[counts < 10].index
        df = df.drop(df[df['job'].isin(under10)].index)

        df['sentence_count'] = df['answer'].apply(lambda x: self.sentence_count_gensim(x))
        df = df.drop(df[df['sentence_count'] < 5].index)

        df['summary'] = df['answer'].apply(lambda x: summarize(x, word_count=50))
        df = df.sort_values('job')
        df = df.drop(df[df['summary'] == ""].index)

        result_df = df[['job', 'summary']]
        result_df.to_csv(self.output_file, index=False, encoding='UTF-8')

job_summary = JobSummary('csv/jobkorea_question_labeled_mecab.csv', '직무별 요약.csv')
job_summary.process_data()
