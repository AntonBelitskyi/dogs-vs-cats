import json
from io import BytesIO

from .testcases import BaseTestCase
from predicts.constants import UNKNOWN_CLASS, CAT, DOG

FIXTURES = {
    'cat_jpeg.jpg': CAT,
    'cat_jpeg_rotate_90.jpg': CAT,
    'cat_jpeg_rotate_180.jpg': CAT,
    'cat_jpeg_rotate_270.jpg': CAT,
    'cat_png.png': CAT,
    'dog_jpeg.jpg': DOG,
    'dog_png.png': DOG,
    'no_dog_and_no_cat_jpeg.jpg': UNKNOWN_CLASS,
    'no_dog_and_no_cat_png.png': UNKNOWN_CLASS,
}


class ApiEndpointsTestCase(BaseTestCase):

    def test_predict_cats_and_dogs_endpoint(self):
        for image, result in FIXTURES.items():
            image_file = open(self.fixtures_dir + image, "rb").read()
            data = {"image": (BytesIO(image_file), image)}
            resp = self.app.post(
                "/api/v1/predict/cats_and_dogs/",
                data=data
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(json.loads(resp.data)["data"], result)

    def test_wrong_predict_model_name(self):
        image = list(FIXTURES)[0]
        image_file = open(self.fixtures_dir + image, "rb").read()
        data = {"image": (BytesIO(image_file), image)}
        resp = self.app.post(
            "/api/v1/predict/wrong-predict-model-name/",
            data=data
        )
        self.assertEqual(resp.status_code, 400)

    def test_make_request_without_image(self):
        data = {}
        resp = self.app.post(
            "/api/v1/predict/cats_and_dogs/",
            data=data
        )
        self.assertEqual(resp.status_code, 400)

    def test_send_image_with_unsupported_mime_type(self):
        image = "no_jpeg_no_png.svg"
        image_file = open(self.fixtures_dir + image, "rb").read()
        data = {"image": (BytesIO(image_file), image)}
        resp = self.app.post(
            "/api/v1/predict/cats_and_dogs/",
            data=data
        )
        self.assertEqual(resp.status_code, 400)
