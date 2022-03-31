from ml_app import app
from pathlib import Path
import os

basedir = Path(__file__).parent.parent.parent
filepath = os.path.join(basedir, 'iris.csv')


def test_create_endpoint():
    """
    GIVEN a Flask application 
    WHEN the '/create' endpoint is requested (POST)
    THEN check that the response is valid
    """

    with app.test_client() as test_client:
        response = test_client.post(
            '/create',
            query_string={'target': 'Species'},
            data={
                'csv_file': open(filepath, 'rb'),
            }
        )
        assert response.status_code == 200


def test_target_variable_in_request():
    """
    GIVEN a Flask application 
    WHEN the '/create' endpoint is requested (POST)
    without the target variable
    It should throw 404 error
    """

    with app.test_client() as test_client:
        response = test_client.post(
            '/create',
            query_string={},
            data={
                'csv_file': open(filepath, 'rb'),
            }
        )
        assert response.status_code == 400


def test_dataset_in_request():
    """
    GIVEN a Flask application 
    WHEN the '/create' endpoint is requested (POST)
    without the dataset
    It should throw 404 error
    """

    with app.test_client() as test_client:
        response = test_client.post(
            '/create',
            query_string={'target': 'Species'},
            data={}
        )
        assert response.status_code == 400
