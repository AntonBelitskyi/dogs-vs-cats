from .predict_models import *


def get_prediction_model(predict_model):
    """
    Getting predict model class that bases on BasePredictModel
    :param predict_model: str
    :return: PredictModel
    """
    predict_model_class = [model for model in BasePredictModel.__subclasses__() if model.NAME == predict_model]
    return predict_model_class[0] if predict_model_class else None
