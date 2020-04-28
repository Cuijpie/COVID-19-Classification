from models.googleNet import GoogleNet


def main() -> None:
    model = GoogleNet()
    model.model.summary()



if __name__ == "__main__":
    main()
