from models.googleNet import GoogleNet
from models.denseNet import DenseNet

def main() -> None:
    model = DenseNet()

    model.model.summary()


if __name__ == "__main__":
    main()
