import pandas as pd
from gensim.summarization import summarize
import warnings
warnings.filterwarnings('ignore')

class QuestionSummary:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def sentence_count_gensim(self, text):
        from gensim.summarization.textcleaner import split_sentences
        return len(split_sentences(text))

    def process_data(self):
        df = pd.read_csv(self.input_file)

        df['sentence_count'] = df['answer'].apply(lambda x: self.sentence_count_gensim(x))
        df = df.drop(df[df['sentence_count'] < 5].index)

        questions = ['지원동기', '입사 후 포부', '성장과정', '성격 장단점', '직무역량', '성취경험', '협력경험', '인생목표', '갈등경험', '자기소개', '직무 및 기업 이해도']
        result_df = pd.DataFrame(columns=['question', 'summary'])

        for question in questions:
            question_df = df[df[question] == 1]
            question_df['summary'] = question_df['answer'].apply(lambda x: summarize(x, word_count=30))
            question_df = question_df.drop(question_df[question_df['summary'] == ""].index)

            for index, row in question_df.iterrows():
                result_df = result_df.append({'question': question, 'summary': row['summary']}, ignore_index=True)

        result_df.to_csv(self.output_file, index=False, encoding='UTF-8')

question_summary = QuestionSummary('csv/jobkorea_question_labeled_mecab.csv', '문항별 요약.csv')
question_summary.process_data()
