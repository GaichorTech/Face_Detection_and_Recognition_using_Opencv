# FACE DETECTION AND RECOGNITION

This project uses the opencv library to train a facial recognition model and use it to verify the user later using the trained model.

## HOW TO RUN THE PROJECT
### 1.Create a python virtual environment and activate it
> python -m venv venv
> source ~/venv/bin/activate

### 2. Install the required libraries using the requirements file
> pip install -r requirements.txt

### 3. Run the collect_data.py file. This will gather samples of your face. Wait till 200 samples are collected
> python collect_data.py

### 4. Run the facial_recognition.py file. This will train a model using your face samples and later use it to recognize your face.
> python facial_recognition.py 