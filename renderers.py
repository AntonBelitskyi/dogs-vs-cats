from flask_api.renderers import JSONRenderer


class JSONAPIRenderer(JSONRenderer):
    """ To accept requests made with the 'application/vnd.api+json' content-type """

    media_type = "application/vnd.api+json"
