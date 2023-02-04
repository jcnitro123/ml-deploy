from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "D-DOC API RF MODEL"


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    symptoms_1 = request.form.get('symptoms_1')
    symptoms_2 = request.form.get('symptoms_2')
    symptoms_3 = request.form.get('symptoms_3')
    symptoms_4 = request.form.get('symptoms_4')
    symptoms_5 = request.form.get('symptoms_5')
    symptoms_6 = request.form.get('symptoms_6')
    symptoms_7 = request.form.get('symptoms_7')
    symptoms_8 = request.form.get('symptoms_8')
    symptoms_9 = request.form.get('symptoms_9')

    input_query = np.array([[age, gender, symptoms_1, symptoms_2, symptoms_3,
                             symptoms_4, symptoms_5, symptoms_6, symptoms_7,
                             symptoms_8, symptoms_9]])

    predictions = model.predict_proba(input_query)[0]
    top3_indexes = predictions.argsort()[-3:][::-1]
    top3_results = [model.classes_[i] for i in top3_indexes]

    return jsonify({'disease': str(top3_results)})


if __name__ == '__main__':
    app.run(debug=True)
