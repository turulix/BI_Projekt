import pathlib
import pickle
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.pyplot import figure
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from training.misc import get_train_data


def main():
    train_data = pd.read_csv("../data/atp_matches_till_2022_with_career.csv")
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    x_train, y_train, x_test, y_test = get_train_data(train_data, test_data)
    if pathlib.Path("../data/model_ensemble.pkl").exists():
        gradient_model = pickle.load(open("../data/model_ensemble.pkl", "rb"))
    else:
        gradient_model = GradientBoostingClassifier(
            learning_rate=0.013713350534884634,
            max_depth=8,
            max_features=86,
            min_samples_split=14,
            n_estimators=1360,
        )
        gradient_model.fit(x_train, y_train)
        print(gradient_model.score(x_test, y_test))
        pickle.dump(gradient_model, open("../data/model_ensemble.pkl", "wb"))

    predictions = gradient_model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Player 1", "Player 2"])
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["figure.dpi"] = 200
    disp.plot()
    plt.show()
    print(cm)


if __name__ == "__main__":
    main()
