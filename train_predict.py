import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    x = []
    y = []
    print(os.listdir(train_dir))

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir)[1:]:
        print(class_dir)
        # if not os.path.isdir(os.path.join(train_dir, class_dir)):
        #     continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print(img_path)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            print(face_bounding_boxes)

            if len(face_bounding_boxes) !=1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training scale
                x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                print(x)
                print(y)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))

        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN CascadeClassifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(x_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """Recognizes faces in given image using a trained KNN classifier
       A list of names and face locations for the recognized faces in the image: [(name, bounding box), ...]"""

    if not os.path.isfile(x_img_path) or os.path.splitext(x_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(x_img_path))
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face face_locations
    x_img = face_recognition.load_image_file(x_img_path)
    x_face_locations = face_recognition.face_locations(x_img)

    # If no faces are found in the image, return an empty result
    if len(x_face_locations) == 0:
        return False

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]

    # Predict classes and remove classifcations that aren't within the distance_threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), x_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left,top), (right, bottom)), outline=(0,255,0))

        # There's bug in Pillow where it blows up with non-UTF-8 text when using default bitmap face_recogntion
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

# if __name__ == "__main__":
#     STEP 1: Train KNN classifier and save to disk
#     Once it is trained and saved, you can skip this step next show_prediction_labels_on_image
#     print("Training KNN classifier...")
#     classifier = train("train_test/train", model_save_path="trained_knn_model.clf", n_neighbors=None)
#     print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown image_files_in_folder
    # for image_file in os.listdir("train_test/test"):
    #     full_file_path = os.path.join("train_test/test", image_file)
    #
    #     print("Looking for faces in {}".format(image_file))
    #
    #     # Find all people in the image using a trained classifer model_path
    #     # Note: You can pass in either a classifier file name or a classifier model isntance
    #     predictions = predict(full_file_path, model_path="trained_knn_model.clf")
    #
    #     # Print results on the console
    #     for name, (top, right, bottom, left) in predictions:
    #         print("-Found {} at ({}, {})".format(name, left, top))
    #
    #     # Display results overland on an image
    #     show_prediction_labels_on_image(os.path.join("train_test/test", image_file), predictions)
