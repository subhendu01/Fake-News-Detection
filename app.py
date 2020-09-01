# https://medium.com/analytics-vidhya/building-a-fake-news-classifier-deploying-it-using-flask-6aac31dfe31d
import os
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
from sklearn import *
import pickle
import flask
import newspaper
from newspaper import Article
import urllib

# start flask
app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

app = flask.Flask(__name__, template_folder='templates')

with open('model/model.pickle', 'rb') as handle:
    model = pickle.load(handle)

# render default webpage
@app.route('/')
def main():
    return render_template('index.html')

# when the post method detect, then redirect to success function
# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    # print(news)
    # Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)