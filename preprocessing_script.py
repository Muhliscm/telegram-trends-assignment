import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stopwords.words("english")


class PreprocessingPipeline:
    """sumary_line

    take column from data as input and return the data after stemming

    Keyword arguments:
    argument -- data frame column
    Return: preprocessed data
    """

    def __init__(self, data) -> None:
        self.data = data

    @staticmethod
    def stemming(text):

        # lower text
        text = str(text).lower()
        # convert into words
        text = nltk.word_tokenize(text)

        # removing special characters
        y = []
        for str_ in text:
            if str_.isalnum():
                y.append(str_)

        text = y.copy()
        y.clear()

        # removing stop words and punctuation
        for str_ in text:
            if str_ not in stopwords.words('english') and str_ not in string.punctuation:
                y.append(str_)

        text = y.copy()
        y.clear()

        # stemming
        ps = PorterStemmer()
        for str_ in text:
            y.append(ps.stem(str_))

        return " ".join(y)

    def get_stemmed_data(self):
        return self.data.apply(PreprocessingPipeline.stemming)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("Data_Science_interview - Data_Science_interview.csv")
    pipeline = PreprocessingPipeline(df["Message"])
    df["stemmed data"] = pipeline.get_stemmed_data()
    print(df['stemmed data'].sample(5))
