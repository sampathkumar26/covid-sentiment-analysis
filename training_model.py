#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer # for text vectorizing

from sklearn.metrics import roc_auc_score 

# for trainig and saving th model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import pickle

#  librariesfor cleaning the text
import neattext as nt
import neattext.functions as nfx
import re
import string

#importing the training data
traindf = pd.read_csv("training.csv")

traindf['Labels'] = traindf['Labels'].apply(lambda x: [int(i) for i in x.split()] )

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
    "joking": 10
}

# this function we are using for creating/ adding the columns and populating them on the basis of the labels
for k,v in class_map.items():
    traindf[k]=traindf['Labels'].apply(lambda x: 1 if v  in x else 0)
    traindf[k]=traindf[k].astype(float) # converting the numbers into float

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
traindf['Tweet'] = traindf['Tweet'].apply(lambda x:clean_text(x))

# now using neattext for removing the stopwards from the tweets which are creating noise in the data

traindf['Tweet'].apply(lambda x:nt.TextFrame(x).noise_scan())
traindf['Tweet'].apply(lambda x:nt.TextExtractor(x).extract_stopwords())
traindf['Tweet'].apply(nfx.remove_stopwords)
traindf['Tweet']= traindf['Tweet'].apply(nfx.remove_stopwords)

# Dividing the data into input and output variables
X =  traindf.Tweet
y =  traindf.drop(['ID','Labels','Tweet'],axis = 1)

# defining the word vectorizier for converting the tweets into vectors
word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',     
    analyzer='word',            
    token_pattern=r'\w{1,}',    
    ngram_range=(1, 3),         
    stop_words='english',
    sublinear_tf=True)

word_vectorizer.fit(X)    

train_word_features = word_vectorizer.transform(X)

#Saving vectorizer
pickle.dump(word_vectorizer, open("vectorizer.pkl", "wb"))

# defining the classifier here we are using SGDclassifier 
# Stochastic Gradient Descent (SGD) is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of  functions that minimize a cost function.
# OneVsRestClassifier is the heuristic method for using binary classification algorithms for multi-class classification

classifier = OneVsRestClassifier(SGDClassifier(random_state=0,loss='log',alpha=0.00001,penalty='elasticnet'))
classifier.fit(train_word_features, y.values)

y_train_pred_proba = classifier.predict_proba(train_word_features)
pickle.dump(classifier, open("model.sav", 'wb'))

roc_auc_score_train = roc_auc_score(y, y_train_pred_proba,average='weighted')

print(roc_auc_score_train)