import os
from joblib import load
from flask import Flask, send_file, request, render_template
from ga import GeneticAlgorithm

app = Flask(__name__, template_folder='template')

# Initialize GeneticAlgorithm instance
ga_instance = GeneticAlgorithm(
    model_path='./best_model.joblib',
    dictionary_path='./dictionary.txt'
)


@app.route("/")
def index():
    return send_file('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        classifier = load('./best_model.joblib')
        sentence = request.form['sentence']
        prediction = classifier.predict([sentence])
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')


@app.route("/generate", methods=['GET'])
def generate():
    generated_results = ga_instance.run()
    return render_template('generate.html', generated_results=generated_results)


@app.route("/predict_multiple", methods=['GET', 'POST'])
def predict_multiple():
    if request.method == 'POST':
        classifier = load('./best_model.joblib')
        sentences = request.form.getlist('sentence')
        predictions = [classifier.predict([sentence])[0] for sentence in sentences]
        average_prediction = sum(predictions) / len(predictions)
        return render_template('predict_multiple.html', predictions=predictions, average_prediction=average_prediction)
    return render_template('predict_multiple.html')


def main():
    app.run(port=3000, debug=True, host='0.0.0.0')


if __name__ == "__main__":
    main()
