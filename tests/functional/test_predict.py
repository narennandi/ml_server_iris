from ml_app import app
from pathlib import Path
import os
import json


def test_predict_endpoint():
    """
    GIVEN a Flask application 
    WHEN the '/predict' endpoint is requested (POST)
    THEN check that the response is the right output
    """

    with app.test_client() as test_client:
        response = test_client.post(
            '/predict',
            data={
                'input_line': '5.1,3.5,1.4,0.2'
            }
        )
        assert response.data == b'setosa'


def test_loading_saved_model():
    """
    GIVEN a Flask application 
    WHEN the '/predict' endpoint is requested (POST)
    WITH an incorrect model name to load from
    THEN check that it returns 404
    """

    with app.test_client() as test_client:
        response = test_client.post(
            '/predict',
            query_string={'filename': 'svm_model_fake.sav'},
            data={
                'input_line': '5.1,3.5,1.4,0.2'
            }
        )
        assert response.status_code == 404
