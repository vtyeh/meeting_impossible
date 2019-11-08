# Import Dependencies for Flask and MongoDB
import os
import pandas as pd
import random
from flask import render_template, request, flash, redirect, Flask
import pymongo
from pymongo import MongoClient
from flask_pymongo import PyMongo
from config import DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASS

# Import Dependencies for Face Recognition
import cv2
import face_recognition
from PIL import Image 

# Import Trained KNN Model
from train_predict import train, predict, show_prediction_labels_on_image

# Import SignUp form
from forms import SignUp

# Connect to MongoDB
# retryWrites: Whether supported write operations executed within this MongoClient will be retried once after a network error on MongoDB 3.6+.
mng_client = pymongo.MongoClient(DB_HOST, DB_PORT, retryWrites=False) 
db = mng_client[DB_NAME]
db.authenticate(DB_USER, DB_PASS)

# mng_client = pymongo.MongoClient('localhost', 27017)
# db = mng_client['app']

# Create app
app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def sign_up():
    """ Save new user info into Mongo Database """
    form = SignUp()

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
        new_entry["Description"] = description
        new_entry["Unique ID"] = unique_id

        # Create new collection for users in app database and insert data
        collection = "users"
        db_cm = db[collection]
        data = new_entry.to_dict(orient='records')
        db_cm.insert_one(data[0])

        try: 
            # Retrain model with new users
            print("Training KNN classifier...")
            train("static/train_test/train", model_save_path="static/models/trained_knn_model.clf", n_neighbors=None)
            print("Training complete!")
        except:
            print("Oh no, unable to train model.")

        status="Agent assignment complete."
        return render_template("index.html", form=form, status=status)
    else:
        return render_template("index.html", form=form)


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
    return render_template("search.html")

# Allowed extensions for uploaded images.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def acquaintence_identification(uploaded_file):
    status_message = "Sorry, unable to identify them :( Please try another photo."

    print("Looking for faces in {}".format(uploaded_file))

    # Find all people in the image using a trained classifer model_path
    # Note: You can pass in either a classifier file name or a classifier model instance

    predictions = predict(uploaded_file, model_path="static/models/trained_knn_model.clf")

    if predictions == False:
        return render_template("search.html", status=status_message)

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:

        print("-Found {} at ({}, {})".format(name, left, top))

        if name == "unknown":
            return render_template("search.html", status=status_message)
        else:
            collection = db.users
            query = collection.find({"Unique ID": name})
            user_dict = {}
            for q in query:
                user_dict['Name'] = q['Name']
                user_dict['Phone'] = q['Phone']
                user_dict['Email'] = q['Email']
                user_dict['Occupation'] = q['Occupation']
                user_dict['Description'] = q['Description']
            print("Rendering template...")
            return render_template("search.html", status="No one here.", user=user_dict, src_path=uploaded_file )
               
    # Display results overland on an image
    # show_prediction_labels_on_image(uploaded_file, predictions)

if __name__ == "__main__":
    app.run(debug=True)
