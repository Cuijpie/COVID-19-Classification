def create_model():
    return "A model"


class GoogleNet(object):
    def __init__(self):
        self.model = create_model()

    def summary(self):
        return print(self.model)

