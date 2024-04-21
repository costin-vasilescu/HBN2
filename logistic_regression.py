from joblib import load

class LR:
    def __init__(self, model_path):
        # Load the trained pipeline
        self.pipeline = load(model_path)

    def predict(self, input_text):
        # Predict using the loaded pipeline, which handles vectorization automatically
        prediction = self.pipeline.predict([input_text])
        return prediction