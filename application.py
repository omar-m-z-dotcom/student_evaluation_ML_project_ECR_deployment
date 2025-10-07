from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)
app = application


@app.route("/", methods=["GET"])
def index():
    """
    This function renders the index.html template when the root URL of the application is accessed.

    Returns:
        str: The rendered index.html template.
    """
    return render_template("index.html")


@app.route("/predict_datapoint", methods=["GET", "POST"])
def predict_datapoint():
    """
    This function is used to predict the math score of a student based on their other exam scores and demographic information.

    If the request method is GET, it renders the home.html template. If the request method is POST, it creates a CustomData object with the input data from the form, gets the data as a pandas DataFrame, predicts the math score using the PredictPipeline, logs the input data and the result, and renders the home.html template with the result.

    Returns:
        str: The rendered home.html template.
    """
    try:
        if request.method == "GET":
            return render_template("home.html")
        elif request.method == "POST":
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get(
                    "parental_level_of_education"
                ),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=request.form.get("reading_score"),
                writing_score=request.form.get("writing_score"),
            )
            pred_df = data.get_data_as_dataframe()
            logging.info(f"Predicting for {pred_df}")
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Results: {results[0]}")
            return render_template("home.html", results=results[0])
    except Exception as e:
        logging.exception(e)
        raise CustomException(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
