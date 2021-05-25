import numpy as np
import pandas as pd
from scipy import sparse
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Lemmatizer(BaseEstimator):
    
    ####init method to initiliaze WordNetLemmatizer when obj of class is created
    def __init__(self):
        self.l = WordNetLemmatizer()
    
    #### method to make this class compatible with sklearn pipeline
    def fit(self,x,y=None):
        return self
    
    #### method to transform text into lemmatized tokens
    def transform(self,x):
        x = map(lambda m: ' '.join([self.l.lemmatize(i.lower()) for i in m.split()]),x)
        x = np.array(list(x))
        return x


######----#######------##########------#######-------

def text_clean(comment,tokenizer,lematizer,APPO):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case 
    comment=comment.lower()
    
    #remove \n
    comment=re.sub("\\n","",comment)
    
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup :Aphost lookup dict)
    
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lematizer.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in set(stopwords.words("english"))]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)



######----#######------##########------#######-------

def build_pipeline(x,y,cv=5,scoring='roc_auc'):
    
    pipe = Pipeline(steps = [('lemmatize',Lemmatizer()),
                         ('tfidf_wc',
                          FeatureUnion([
                             ('tfidf_word',TfidfVectorizer(max_features=2500,analyzer='word')),
                             ('tfidf_char',TfidfVectorizer(max_features=2500,analyzer='char'))])),
                 ('logreg',LogisticRegression(solver='liblinear'))]
        )
    
    param_grid = {'tfidf_wc__tfidf_word__max_features':[2500,3000],
                  'tfidf_wc__tfidf_char__max_features':[2500,3000],
                  'logreg__C':[0.1,0.5,1]}
    
    grid_pipe = RandomizedSearchCV(estimator = pipe, param_distributions= param_grid,scoring=scoring, cv =cv )
    grid_pipe.fit(x,y)
    
    return grid_pipe


######----#######------##########------#######-------

def make_prediction(model,xtest,pred_df,class_name):
    
    y_pred = np.round(model.predict_proba(xtest)[:,1],2)
    pred_df[class_name] = y_pred
    return pred_df


    
    
    