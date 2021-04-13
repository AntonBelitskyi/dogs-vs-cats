import numpy as np
from keras.applications import imagenet_utils, ResNet50
from keras_preprocessing.image import img_to_array


class BasePredictModel:
    """
    Base interface to work with prediction models
    """
    NAME = "model name"
    MODEL_PATH = None

    def __init__(self):
        self.model = None

    def init_model(self):
        """Prediction model initialization"""
        return NotImplementedError("You should implement this method")

    def predict(self, image):
        """Run image predicting"""
        return NotImplementedError("You should implement this method")


class CatsAndDogsModel(BasePredictModel):
    """
    Prediction model to determine cats or dogs
    """
    NAME = "cats_and_dogs"
    DOG_CLASSES_PATH = "predicts/cat_dog_labels/dog_classes.txt"
    CAT_CLASSES_PATH = "predicts/cat_dog_labels/cat_classes.txt"

    def init_model(self):
        self.model = ResNet50(weights="imagenet")

    @staticmethod
    def prepare_image(image):
        """
        Resize the input image and preprocess it
        :param image: PIL.Image
        :return: numpy.ndarray
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def read_dog_cat_labels(path):
        """
        Extracts imagenet collection with dog and cats labels
        :param path: str
        :return: list
        """
        labs = list(open(path))
        labs = [item.split(',') for item in labs]
        labs = [item.strip().replace(' ', '_') for sublist in labs for item in sublist]
        return labs

    def is_dog_or_cat(self, class_label):
        """
        Searching matches in dogs and cats labels
        :param class_label:str
        :return: str
        """
        dog_labs = self.read_dog_cat_labels(self.DOG_CLASSES_PATH)
        cat_labs = self.read_dog_cat_labels(self.CAT_CLASSES_PATH)
        if class_label in dog_labs:
            return "Dog"
        elif class_label in cat_labs:
            return "Cat"
        return "Unknown class"

    def predict(self, image):
        """
        Identify the image, Cat or Dog.
        :param image: PIL.Image
        :return: str
        """
        self.init_model()
        image = self.prepare_image(image)
        res = self.model.predict(image)
        results = imagenet_utils.decode_predictions(res)
        if results:
            # Sort results by score
            results[0].sort(key=lambda x: x[2], reverse=True)
            res = results[0][0][1]
            check_result = self.is_dog_or_cat(res)
            return check_result

