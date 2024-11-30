# create handler work with model
# input audio file, output text  or text in file
# stream audio output and return text output
import torch


class Prediction:
    """
    Class for prediction model audio component input or stream.

    Args:
        model (str): path to model file.
    """

    def __init__(self, model=None):
        self.__model = self.default_model_select() if model is None else model

    def get_model(self):
        return self.__model

    def default_model_select(self):
        return self.__load_model("model/model.pt")

    def __load_model(self, path):
        return torch.load(path)

    def predict(self, audio_component, stream=False):
        if stream:
            return self.__model.predict(audio_component)
