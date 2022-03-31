from flask import Flask, request, Response
import pandas as pd
from io import StringIO
import json
from .model import SVMClassifier
import pickle

app = Flask(__name__)

# SVMClassifier is a class i wrote in order to make the code readable
# and more maintainable. It can be found in model.py
svmc = SVMClassifier()
filename = 'svm_model.sav'


@app.route('/create', methods=['POST'])
def create():
    """This endpoint reads a csv file into a pandas dataframe
    extracts the features and the target 
    and then trains a SVM model

    Returns:
        flask.wrappers.Response: 
            HTTP 400 Bad Request if dataset is not passed
            HTTP 400 Bad Request if target argument is not passed
            HTTP 200 Ok if model is trained successfully
    """
    file = request.files.get('csv_file')
    if not file:
        return Response(
            "Missing dataset",
            status=400,
            mimetype='application/json'
        )

    iris = pd.read_csv(file)
    target = request.args.get('target')
    if not target:
        return Response(
            "Missing <target> as a parameter to the request",
            status=400,
            mimetype='application/json'
        )

    x, y = svmc.get_x_y(iris, target)
    x_train, x_test, y_train, y_test = svmc.split_data(x, y, 0.3)
    svmc.fit(x_train, y_train)
    pickle.dump(svmc.model, open(filename, 'wb'))

    return Response(
        None,
        status=200,
        mimetype='application/json'
    )


@app.route('/predict', methods=['POST'])
def predict():
    """This endpoint takes in a single csv line generates a 
    prediction based on the SVM classifier it has stored on disk.

    Returns:
        flask.wrappers.Response:             
            HTTP 200 Ok and the prediction 
            HTTP 404 Not Found if pickled model cannot be found
    """
    data = pd.read_csv(
        StringIO(request.form.get("input_line")),
        header=None
    )
    data = data.to_numpy()
    try:
        svmc.load_model(request.args.get('filename') or filename)
        prediction = svmc.predict(data)
        return Response(
            prediction,
            status=200,
            mimetype='application/json'
        )
    except FileNotFoundError:
        return Response(
            f'Model {filename} not found',
            status=404,
            mimetype='application/json'
        )


# Run the Flask server
if(__name__ == '__main__'):
    app.run(debug=True)
