import logging
from flask import Flask, render_template, request
from model import EnFrTranslator

app = Flask(__name__)
# define model path
model_path = '2BiLSTM.h5'

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
    app.run(port=8000, debug=False) #host="0.0.0.0",

if __name__ == "__main__":
    main()