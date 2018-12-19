from flask import Flask,render_template,url_for,request,redirect
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))
cv = CountVectorizer()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train', methods=['GET'])
def train():
    data = pd.read_csv('spam.csv', encoding="latin-1")
    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
    data['label'] = data['v1'].map({'ham':0,'spam':1})
    data['message'] = data['v2']
    # Allocate data
    x = data['message']
    y = data['label']
    # Fit the data
    x = cv.fit_transform(x)
    # Split data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    # NB Classifier training
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    # Test prediction
    from sklearn.metrics import classification_report
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    # Save the training model to file
    joblib.dump(clf, 'spam_model.pkl')
    # Redirect
    return redirect("http://localhost:5000/", code=302)

@app.route('/predict', methods=['POST'])
def predict():
    # Load trained model
    NB_spam_model = open('spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    # Predict with message
    if request.method == 'POST':
		message = request.form['message']
		data = [message]
        
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)