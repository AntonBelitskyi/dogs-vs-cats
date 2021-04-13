from flask_api import FlaskAPI


def create_app():
    app = FlaskAPI(__name__)

    # Bring in the blueprints which provide routing
    from api.api_v1.api import api as api_v1

    app.register_blueprint(api_v1, url_prefix="/api/v1")

    @app.after_request
    def set_jsonapi_content_type(response):
        if response.headers["Content-Type"] == "application/json":
            response.headers["Content-Type"] = "application/vnd.api+json"
        return response

    return app
