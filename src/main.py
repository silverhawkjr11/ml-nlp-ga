import os
from joblib import load
from flask import Flask, send_file
from flask import request, render_template
from ga import *
app = Flask(__name__, template_folder='template')

@app.route("/")

def index():
    return send_file('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        classifier = load('best_model.joblib')
        sentence = request.form['sentence']
        prediction = classifier.predict([sentence])
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')
@app.route("/generate", methods=['GET'])
def generate():
    generated_results = ga()
    return render_template('generate.html', generated_results=generated_results)

def main():
    app.run(port=int(os.environ.get('PORT', 80
)))

if __name__ == "__main__":
    main()
