import io

import flask
from PIL import Image
from flask import abort, make_response
from flask_api.decorators import set_renderers
from flask_api.renderers import JSONRenderer
from flask_apiblueprint import APIBlueprint

from predicts import get_prediction_model
from renderers import JSONAPIRenderer
from utils import check_mimetype, WrongMimeTypeError

api = APIBlueprint("api_v1", __name__)


@api.route("/predict/<string:prediction_model>/", methods=["POST"])
@set_renderers(JSONAPIRenderer, JSONRenderer)
def predict(prediction_model):
    predict_model = get_prediction_model(prediction_model)
    if predict_model:
        image = flask.request.files["image"]
        if flask.request.files.get("image"):
            try:
                # Check image extension (it should be JPEG or PNG)
                check_mimetype(flask.request.files.get("image"))
            except WrongMimeTypeError:
                return abort(400, "Unsupported file")
            image.seek(0)
            image = image.read()
            image = Image.open(io.BytesIO(image))
            result = predict_model().predict(image)
            resp = make_response({"data": result}, 200)
            return resp
        return abort(400, "There is no image to process")
    return abort(400, f"Prediction model with the name of the '{prediction_model}' does not exist")


