from pandas import pd
from collections import Counter
from pythainlp.corpus import stopwords as stopwords_th
from pythainlp.tokenize import word_tokenize
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer


class ThaiClassifier():
    
    def __init__(self):
        self.df = pd.read_csv(f"./data/data-th.csv")
        self.df = self.df.sort_values(by=['label'])

        self.df["Tweets"] = self.df.Tweets.map(
            self.remove_URL)  # map(lambda x: remove_URL(x))
        self.df["Tweets"] = self.df.Tweets.map(self.remove_punct)
        self.df["Tweets"] = self.df.Tweets.map(self.remove_stopwords)
        self.counter = self.counter_word(self.df.Tweets)
        self.num_unique_words = len(self.counter)
        self.tokenizer = Tokenizer(num_words=self.num_unique_words)


    def counter_word(self, text_col):
        count = Counter()
       
        for text in text_col.values:
                for word in word_tokenize(text):
                    if word != ' ':
                        count[word] += 1

    def remove_URL(self, text):
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)
    
    def remove_punct(self ,text):
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_stopwords(self, text):
        
        stop = stopwords_th.words("thai")
        filtered_words = [word.lower()
                              for word in word_tokenize(text) if word.lower() not in stop]
        return " ".join(filtered_words)

        