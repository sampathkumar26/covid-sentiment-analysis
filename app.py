from flask import Flask,render_template,redirect,request
#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer # for text vectorizing
# for trainig and saving th model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import pickle
#  librariesfor cleaning the text
import re
import string

app=Flask(__name__)

# importing the vectorizer
tf1 = pickle.load(open("vectorizer.pkl", 'rb'))
# importing the model
loaded_model = pickle.load(open("model.sav", 'rb'))

# defining a function for cleaning the tweets removing some specific words and punctuations
def  clean_text(text):
    text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    
    return text

# defining the definition of each labels in the dictionary format
# this dict we are going to use for creating the extra columns for training over model
class_map = {
    "optimistic": 0,
    "thankful": 1,
    "empathetic": 2,
    "pessimistic": 3,
    "anxious": 4,
    "sad": 5,
    "annoyed": 6,
    "denial": 7,
    "surprise": 8,
    "official_report": 9,
    "joking": 10}

word_vectorizer1 = TfidfVectorizer(
            strip_accents='unicode',     
            analyzer='word',            
            token_pattern=r'\w{1,}',    
            ngram_range=(1, 3),         
            stop_words='english',
            sublinear_tf=True,
            vocabulary = tf1.vocabulary_)

@app.route('/')
def hello():
    return render_template("model.html")

@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit',methods=['POST'])
def submit():
    if request.method == 'POST':
        texts = request.form['query']
        print(texts)
        texts = clean_text(texts)
        text_trans = word_vectorizer1.fit_transform([texts])
        y_pred = loaded_model.predict_proba(text_trans)
        predictions = [k for k,v in dict(zip(class_map.keys(),  y_pred[0] )).items() if v >=0.5]
        try:
            predictions = [k for k,v in dict(zip(class_map.keys(),  y_pred[0] )).items() if v >=0.5]
        except:
            predictions = "No labels ar associated with it"
        predictions = ', '.join(predictions)
    return render_template('model.html',prediction = predictions, query = request.form['query'],)


if __name__ =="__main__":
    app.run()




