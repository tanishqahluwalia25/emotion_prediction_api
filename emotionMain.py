from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

@app.route("/")
def predict():
    
    df=pd.read_csv("text_emotion.csv")
    X1=list(df[df.sentiment.isin(("sadness", "worry", "empty", "hate", "boredom", "anger",  ))].content)
    X2=list(df[df.sentiment.isin(("happiness", "relief", "enthusiasm", "love", "fun",   ))].content)
    Y=["neg" for i in X1]
    Y.extend(["pos" for i in X2])

    X=X1
    X.extend(X2)
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    count_vect = CountVectorizer(max_features = 10000)
    x_train_transformed = count_vect.fit_transform(x_train)
    x_test_trans=count_vect.transform(x_test)
    
    
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression( C=0.50, max_iter=5000,penalty="l2")
    lr.fit(x_train_transformed,y_train)
    text = str(request.args.get('text', default="happy"))
    res={
        "text":text,
        "prediction":str(lr.predict(count_vect.transform([text]))[0])
    }
    return jsonify(res)

app.run(debug=True)