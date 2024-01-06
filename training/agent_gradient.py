import datetime

import pandas as pd
import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from misc import get_train_data, maybe_start_sweep, train_and_evaluate

model_type = "Gradient Boosting"
entity = "hka-ml1"
project = "BI_Project"

# Settings for the WandB Sweep, we are using a grid search here.
sweep_configuration = {
    "method": "bayes",
    "name": f"{model_type} - Predicting Wins",
    "metric": {
        "goal": "maximize",
        "name": "mean_test_score"
    },
    "parameters": {
        #  "model": {"values": ["random_forest", "gradient_boosting"]},
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 100,
            "max": 1000
        },
        "max_depth": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 16
        },
        "max_features": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 320
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.5
        },

        # The following parameters are not used by the model, but are used by the training script.
        "n_folds": {"value": 5},
    },
}

sweep_id = maybe_start_sweep(sweep_configuration, project, entity)


def main():
    match_data = pd.read_csv("../data/atp_matches_till_2022_with_career.csv")

    match_data, test_data = train_test_split(match_data, test_size=0.2, random_state=42)

    time_now = datetime.datetime.now()

    x_train, y_train, x_test, y_test = get_train_data(match_data, test_data)

    print("Processing Data Time:", datetime.datetime.now() - time_now)

    # Initialize wandb.
    run = wandb.init()

    model = GradientBoostingClassifier(
        n_estimators=run.config.n_estimators,
        learning_rate=run.config.learning_rate,
        max_depth=run.config.max_depth,
        max_features=run.config.max_features,
    )

    train_and_evaluate(model, x_train, y_train, x_test, y_test, run)


# Start the WandB sweep agent.
if __name__ == "__main__":
    wandb.agent(sweep_id, main)
