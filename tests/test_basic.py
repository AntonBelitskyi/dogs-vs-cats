import numpy
from unittest import mock
from PIL import Image

from predicts import CatsAndDogsModel, get_prediction_model
from predicts.constants import CAT_AND_DOG, UNKNOWN_CLASS
from tests.testcases import BaseTestCase
from utils import check_mimetype, WrongMimeTypeError

FIXTURES = {
    'cat_jpeg.jpg': "image/jpeg",
    'cat_png.png': "image/png",
}


class CheckMimeTypeTestCase(BaseTestCase):

    def test_check_mimetype_using_filepath(self):
        for image, result in FIXTURES.items():
            image_path = self.fixtures_dir + image
            mime_type = check_mimetype(image_path)
            self.assertEqual(mime_type, result)

    def test_check_mimetype_using_image_object(self):
        for image, result in FIXTURES.items():
            with open(self.fixtures_dir + image, "rb") as image:
                mime_type = check_mimetype(image)
                self.assertEqual(mime_type, result)

    def test_check_mimetype_with_unsupported_type(self):
        image = "no_jpeg_no_png.svg"
        image_path = self.fixtures_dir + image
        with self.assertRaises(WrongMimeTypeError):
            check_mimetype(image_path)


class CatsAndDogsModelTestCase(BaseTestCase):

    def setUp(self):
        super(CatsAndDogsModelTestCase, self).setUp()
        self.predict_model = CatsAndDogsModel()

    def test_prepare_image_with_type_jpeg_png(self):
        for image in list(FIXTURES):
            image_path = self.fixtures_dir + image
            image = Image.open(image_path)
            processed_image = self.predict_model.prepare_image(image)
            self.assertEqual(processed_image.shape, (1, 224, 224, 3))
            self.assertIsInstance(processed_image, numpy.ndarray)

    def test_read_dog_cat_labels(self):
        cat_labels = self.predict_model.read_dog_cat_labels(CatsAndDogsModel.CAT_CLASSES_PATH)
        self.assertEqual(len(cat_labels), 30)
        dog_labels = self.predict_model.read_dog_cat_labels(CatsAndDogsModel.DOG_CLASSES_PATH)
        self.assertEqual(len(dog_labels), 201)

    def test_is_dog_or_cat(self):
        is_cat = self.predict_model.is_dog_or_cat("tabby")
        is_dog = self.predict_model.is_dog_or_cat("Chihuahua")
        unknown_class = self.predict_model.is_dog_or_cat("octopus")
        self.assertEqual(is_cat, CAT_AND_DOG.get("CAT"))
        self.assertEqual(is_dog, CAT_AND_DOG.get("DOG"))
        self.assertEqual(unknown_class, UNKNOWN_CLASS)

    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Dog")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_dog(self, mock_resnet_50, mock_imagenet_utils):
        results = [[
            ("Unknown class", "class_description", 1),
            ("Cat", "class_description", 2),
            ("Dog", "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual("Dog", self.predict_model.predict(mock.Mock()))


    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Cat")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_cat(self, mock_resnet_50, mock_imagenet_utils):
        results = [[
            ("Unknown class", "class_description", 1),
            ("Dog", "class_description", 2),
            ("Cat", "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual("Cat", self.predict_model.predict(mock.Mock()))

    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Unknown class")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_cat(self, mock_resnet_50, mock_imagenet_utils):
        results = [[
            ("Cat", "class_description", 1),
            ("Dog", "class_description", 2),
            ("Unknown class", "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual("Unknown class", self.predict_model.predict(mock.Mock()))


class GetPredictionModelTestCase(BaseTestCase):

    def test_get_prediction_model(self):
        model = get_prediction_model("cats_and_dogs")
        self.assertEqual(model, CatsAndDogsModel)
        self.assertIsNone(get_prediction_model("nonexistent model"))