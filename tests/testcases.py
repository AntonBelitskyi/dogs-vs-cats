import unittest

from api import create_app


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        app = create_app()
        self.app = app.test_client()
        self.fixtures_dir = "tests/fixtures/"
