import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from training.misc import get_train_data


def main():
    train_data = pd.read_csv("../data/dmml1_train.csv")
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    x_train, y_train, x_test, y_test = get_train_data(train_data, test_data)

    gradient_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=10,
        max_features=16,
        learning_rate=0.1,
    )

    gradient_model.fit(x_train, y_train)

    print(gradient_model.score(x_test, y_test))

    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    predictions = gradient_model.predict(x_test)

    with_predictions = pd.concat([x_test, y_test, pd.DataFrame(predictions, columns=["Prediction"])], axis=1)
    with_predictions["Error"] = with_predictions["Prediction"] - with_predictions["Sales"]

    with_predictions["Penalty"] = np.where(with_predictions["Error"] < -4000, 100, 0)
    with_predictions["Penalty"] = np.where(with_predictions["Error"] > 3000, 150, with_predictions["Penalty"])
    with_predictions["Penalty"] = np.where(with_predictions["Error"] > 6000, 250, with_predictions["Penalty"])

    with_predictions = with_predictions[with_predictions["remainder__Open"] == 1]

    total_penalty = with_predictions["Penalty"].sum()
    print(total_penalty)
    mean_penalty = total_penalty / len(with_predictions)
    print(mean_penalty)

    pickle.dump(gradient_model, open("../data/model_ensemble.pkl", "wb"))


if __name__ == "__main__":
    main()
