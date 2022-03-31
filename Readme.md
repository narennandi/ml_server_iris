# To run the app, from the basedir ./ml_server_iris
# open a CMD prompt cd into ./ml_server_iris
# run the below commands in order 

# to start the app, i developed it in python 3.7
python -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt

# To run the app
set FLASK_APP=ml_app
flask run

# if you would like to run tests
python -m pytest

# Postman Collection
I have exported my postman collection. ML_DataRobot.postman_collection.json
It can be directly imported to postman file > import >  ML_DataRobot.postman_collection.json
It consists of two requets. Create and predict.
For the create request in the body you will need to point the value for the csv_file to the correct location

# curl commands, 
# i have tested the below commands in postman by going to 
# file > import > raw text and pasting the below commands as separate requests
# They are the same as the above postman collection
curl -X POST -F 'csv_file=@iris.csv' 'http://localhost:5000/create?target=Species'
curl -X POST -F 'input_line=5.1,3.5,1.4,0.2' 'http://localhost:5000/predict'


# Directory Structure
The app is broken down into
    .venv - The virtual environment folder, running python 3.7
    ml_app - the flask app which contains the project related code
    tests - contains the functional tests for the code
    iris.csv - the iris dataset downloaded from the link in the word doc
    ML_DataRobot.postman_collection.json - Postman collection that consists of the two requests
    requirements.txt - contains all the project requirements
    svm_model.sav - this is the pickle file for the svm model
    
