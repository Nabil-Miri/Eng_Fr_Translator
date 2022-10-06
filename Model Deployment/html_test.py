import os
import logging
from flask import Flask, render_template, request, jsonify
from model import EnFrTranslator

app = Flask(__name__)
# define model path
model_path = './model/2BiLSTM.h5'
# @app.route("/", methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         eng=first_name = request.form.get("en")
#         if request.form.get('submit') == 'Translate':
#             pass predict(eng)
#     elif request.method == 'GET':
#         return render_template('HTML.html')
#     return render_template("HTML.html")
# create instance
model = EnFrTranslator(model_path)
logging.basicConfig(level=logging.INFO)
@app.route("/hello")
def index():
    """Provide simple health check route."""
    return "Hello world!"
@app.route("/", methods=["GET", "POST"])
def translate():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")

    if request.method == 'POST':
        sentence = request.form["text"]
        prediction = model.translate(sentence)
        logging.info("prediction from model= {}".format(prediction))
        return render_template('HTML.html', prediction = prediction)
    else:
        return render_template('HTML.html', prediction = "Enter a Sentence")


def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False)
if __name__ == "__main__":
    main()