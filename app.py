# Import Dependencies for Flask and MongoDB
import os
import pandas as pd
import random
from flask import render_template, request, flash, redirect, Flask
# from flask_caching import Cache
import pymongo
from pymongo import MongoClient
from flask_pymongo import PyMongo
from config import DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASS

# Import Dependencies for Face Recognition
import cv2
import face_recognition
from PIL import Image 

# Import Trained KNN Model and Sign Up Form
from train_predict import train, predict, show_prediction_labels_on_image
from forms import SignUp

# Import Dependencies for Hate Speech Recognition
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize  # tokenizer used when training TFIDF vectorizer
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import pickle
import json
import numpy as np

# Allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Connect to MongoDB
mng_client = pymongo.MongoClient(DB_HOST, DB_PORT)
db = mng_client[DB_NAME]
db.authenticate(DB_USER, DB_PASS)

# mng_client = pymongo.MongoClient('localhost', 27017)
# db = mng_client['app']

# Create app
app = Flask(__name__)
# cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.before_first_request
def nbsvm_models():
     global tfidf_model
     global logistic_identity_hate_model
     global logistic_insult_model
     global logistic_obscene_model
     global logistic_severe_toxic_model
     global logistic_threat_model
     global logistic_toxic_model

# Load all trained hatespeech models
with open('static/models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
    tfidf_model = pickle.load(tfidf_file)
with open('static/models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
    logistic_toxic_model = pickle.load(logistic_toxic_file)
with open('static/models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
    logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)
with open('static//models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
    logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)
with open('static/models/logistic_insult.pkl', 'rb') as logistic_insult_file:
    logistic_insult_model = pickle.load(logistic_insult_file)
with open('static/models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
    logistic_obscene_model = pickle.load(logistic_obscene_file)
with open('static//models/logistic_threat.pkl', 'rb') as logistic_threat_file:
    logistic_threat_model = pickle.load(logistic_threat_file)

@app.route("/", methods = ["GET", "POST"])
def sign_up():
    """ Save new user info into Mongo Database """
    form = SignUp(csrf_enabled=False)

    name = []
    phone = []
    email = []
    occupation = []
    age = []
    description = []
    unique_id = []

    if request.method == "POST":

        name.append(form.name.data)
        phone.append(form.phone.data)
        email.append(form.email.data)
        occupation.append(form.occupation.data)
        age.append(form.age.data)
        description.append(form.description.data)
        unique_id.append(form.unique_id.data)

        # Save new user's image for retraining the face recognition model
        save_image_for_training(unique_id[0], name[0])

        new_entry = pd.DataFrame()

        new_entry["Name"] = name
        new_entry["Phone"] = phone
        new_entry["Email"] = email
        new_entry["Occupation"] = occupation
        new_entry["Age"] = age
        new_entry["Description"] = description
        new_entry["Unique ID"] = unique_id

        # Create new collection for users in app database and insert data
        collection = "users"
        db_cm = db[collection]
        data = new_entry.to_dict(orient='records')
        db_cm.insert(data)

        status="Agent assignment complete."
        return render_template("index.html", form=form, status=status)
    else:
        return render_template("index.html", form=form)

def save_image_for_training(unique_id, name):
    """ Saves image into file """

    # Check if valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

    file = request.files['file']
    print(file)

    if file.filename == '':
        return redirect(request.url)

    # If valid, save the image
    if file and allowed_file(file.filename):
        new_path = f"static/train_test/train/{unique_id}"
        os.makedirs(new_path, exist_ok=True)
        img_path = f"{new_path}/{name}.png"
        im = Image.open(file)
        im.save(img_path)

@app.route('/headquarters', methods=['GET', 'POST'])
def upload_image():
    # Check if valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        print(file)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            random_id = random.randint(0, 10000)
            name = f"static/train_test/test/{random_id}.png"
            im = Image.open(file)
            im.save(name)

            # Detect faces and return the result
            return acquaintence_identification(name)

    # If no valid image file was uploaded, show the file upload form:
    return render_template("search.html", user=None)

def acquaintence_identification(uploaded_file):

    try: 
        # Retrain model with new users
        print("Training KNN classifier...")
        train("static/train_test/train", model_save_path="static/models/trained_knn_model.clf", n_neighbors=None)
        print("Training complete!")
    except:
        return render_template("search.html", status="Oops, no one is in here. Be the first to sign up! (:")

    print("Looking for faces in {}".format(uploaded_file))

    # Find all people in the image using a trained classifer model_path
    # Note: You can pass in either a classifier file name or a classifier model isntance
    predictions = predict(uploaded_file, model_path="static/models/trained_knn_model.clf")

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("-Found {} at ({}, {})".format(name, left, top))
        if name == "unknown":
            return render_template("search.html", status="Sorry, unable to identify them :( Please try another photo.")
        else:
            collection = db.users
            query = collection.find({"Unique ID": name})
            for q in query:
                user_dict = {}
                user_dict['Name'] = q['Name']
                user_dict['Phone'] = q['Phone']
                user_dict['Email'] = q['Email']
                user_dict['Occupation'] = q['Occupation']
                user_dict['Description'] = q['Description']
            print(user_dict)
               
    # Display results overland on an image
    # show_prediction_labels_on_image(uploaded_file, predictions)
    
    print("Rendering template...")
    return render_template("search.html", status="No one here.", user=user_dict, src_path=uploaded_file )


@app.route('/hatespeech', methods=['GET', 'POST'])
def my_form_post():
    """ Takes the comment submitted by the user, apply TFIDF trained vectorizer to it, predict using trained models """
    if request.method == 'POST':

        text = request.form['text']
        temp = []
        temp.append(request.form['text'])
        length = len(temp[0].split())

        comment_term_doc = tfidf_model.transform([text])

        dict_preds = {}

        dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
        dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
        dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
        dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
        dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
        dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]

        for k in dict_preds:
            perc = dict_preds[k] * 100
            dict_preds[k] = "{0:.2f}%".format(perc)
            if length > 1:
                result = get_emoji(temp)
                inx1 = int(result[2])
                inx2 = int(result[3])
                inx3 = int(result[4])
                inx4 = int(result[5])
                value = str(result[0])
                emoji_result = emo[inx1]+emo[inx2]+emo[inx3]+emo[inx4]
                print(value)
                  
        return render_template('hatespeech.html', text=text, result=value, emoji_result=emoji_result,
                            pred_toxic=dict_preds['pred_toxic'],
                            pred_severe_toxic=dict_preds['pred_severe_toxic'],
                            pred_identity_hate=dict_preds['pred_identity_hate'],
                            pred_insult=dict_preds['pred_insult'],
                            pred_obscene=dict_preds['pred_obscene'],
                            pred_threat=dict_preds['pred_threat'])
    else:
        return render_template('hatespeech.html')

# Add emojis based on models
maxlen = 30

emo = ['😂', '😒', '😩', '😭', '😍',
       '😔', '👌', '😊', '❤', '😏',
       '😁', '🎶', '😳', '💯', '😴',
       '😌', '☺', '🙌', '💕', '😑',
       '😅', '🙏', '😕', '😘', '♥',
       '😐', '💁', '😞', '🙈', '😫',
       '✌', '😎', '😡', '👍', '😢',
       '😪', '😋', '😤', '✋', '😷',
       '👏', '👀', '🔫', '😣', '😈',
       '😓', '💔', '💓', '🎧', '🙊',
       '😉', '💀', '😖', '😄', '😜',
       '😠', '🙅', '💪', '👊', '💜',
       '💖', '💙', '😬', '✨']

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
# model.summary()
model._make_predict_function()

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

def model_predict(TEST_SENTENCES):
    print(TEST_SENTENCES)
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    print (tokenized)
    prob = model.predict_function([tokenized])[0]
    return prob

def get_emoji(TEST_SENTENCES):
    scores = []
    t_score = []

    print (TEST_SENTENCES)

    prob = model_predict(TEST_SENTENCES)
    t_score.append(TEST_SENTENCES[0])
    t_prob = prob[0]
    ind_top = top_elements(t_prob, 4)

    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)

    print(t_score)
    return t_score

@app.route("/voice-text-voice")
def record():
    return render_template("voice.html")

if __name__ == "__main__":
    app.run(debug=True)
