import numpy
from unittest import mock
from PIL import Image

from predicts import CatsAndDogsModel, get_prediction_model
from predicts.constants import UNKNOWN_CLASS, CAT, DOG
from tests.testcases import BaseTestCase
from tests.test_api import FIXTURES as API_FIXTURES
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

    def test_is_dog_or_cat(self):
        is_cat = self.predict_model.is_dog_or_cat("tabby")
        is_dog = self.predict_model.is_dog_or_cat("Chihuahua")
        unknown_class = self.predict_model.is_dog_or_cat("octopus")
        self.assertEqual(is_cat, CAT)
        self.assertEqual(is_dog, DOG)
        self.assertEqual(unknown_class, UNKNOWN_CLASS)

    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Dog")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_dog(self, mock_resnet_50, mock_imagenet_utils):
        self.predict_model = CatsAndDogsModel()
        results = [[
            (UNKNOWN_CLASS, "class_description", 1),
            (CAT, "class_description", 2),
            (DOG, "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual(DOG, self.predict_model.predict(mock.Mock()))


    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Cat")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_cat(self, mock_resnet_50, mock_imagenet_utils):
        self.predict_model = CatsAndDogsModel()
        results = [[
            (UNKNOWN_CLASS, "class_description", 1),
            (DOG, "class_description", 2),
            (CAT, "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual(CAT, self.predict_model.predict(mock.Mock()))

    @mock.patch("predicts.predict_models.CatsAndDogsModel.is_dog_or_cat", lambda *args: "Unknown class")
    @mock.patch("predicts.predict_models.CatsAndDogsModel.read_dog_cat_labels", mock.Mock())
    @mock.patch("predicts.predict_models.CatsAndDogsModel.prepare_image", mock.Mock())
    @mock.patch("predicts.predict_models.imagenet_utils")
    @mock.patch("predicts.predict_models.ResNet50")
    def test_model_predict_return_unknown_class(self, mock_resnet_50, mock_imagenet_utils):
        self.predict_model = CatsAndDogsModel()
        results = [[
            (CAT, "class_description", 1),
            (DOG, "class_description", 2),
            (UNKNOWN_CLASS, "class_description", 3)
        ]]
        mock_resnet_50.return_value.predict.return_value = mock.Mock()
        mock_imagenet_utils.decode_predictions.return_value = results
        self.assertEqual(UNKNOWN_CLASS, self.predict_model.predict(mock.Mock()))

    def test_model_predict_real_case(self):
        for image_name, result in API_FIXTURES.items():
            image_path = self.fixtures_dir + image_name
            image = Image.open(image_path)
            res = self.predict_model.predict(image)
            self.assertEqual(res, result)


class GetPredictionModelTestCase(BaseTestCase):

    def test_get_prediction_model(self):
        model = get_prediction_model("cats_and_dogs")
        self.assertEqual(model, CatsAndDogsModel)
        self.assertIsNone(get_prediction_model("nonexistent model"))
