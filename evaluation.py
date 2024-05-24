import os
import csv
import tensorflow as tf

def preprocess_image(image_path):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Expand dimensions to match the expected shape
    image_array = tf.expand_dims(image_array, axis=0)

    return image_array

def get_predicted_class(prediction):
    # Get the index of the class with the highest probability
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]

    # Define the class labels
    class_labels = [0, 1]

    # Get the predicted class
    predicted_class = class_labels[predicted_class_index]

    return predicted_class


def predict_class(image_path):
    # Load the model
    model = tf.keras.models.load_model("glomeruli_model.h5")
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Make the prediction
    prediction = model.predict(image)
    
    # Get the predicted class
    predicted_class = get_predicted_class(prediction)
    
    return predicted_class

def evaluate_folder(folder_path):
    # Create the evaluation.csv file
    with open("evaluation.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "predicted_class"])

        # Iterate over the image files in the folder
        for filename in os.listdir(folder_path + "/globally_sclerotic_glomeruli"):
            gsg_path = os.path.join(folder_path, "globally_sclerotic_glomeruli")
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(gsg_path, filename)
                predicted_class = predict_class(image_path)
                writer.writerow([filename, predicted_class])
        for filename in os.listdir(folder_path + "/non_globally_sclerotic_glomeruli"):
            nsg_path = os.path.join(folder_path, "non_globally_sclerotic_glomeruli")
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(nsg_path, filename)
                predicted_class = predict_class(image_path)
                writer.writerow([filename, predicted_class])

# Example usage: evaluate the folder containing the glomeruli image patches
folder_path = "public"

evaluate_folder(folder_path)