import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    x=[]
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            x.append(ps.stem(i))
    return " ".join(x)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the SMS message")

if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    vector = tfidf.transform([transform_sms])
    result = model.predict(vector)

    if result == 1:
        st.header("It's a Spam message")
    else:
        st.header("It's not a Spam message")