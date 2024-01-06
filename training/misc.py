import os
import pathlib
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

import wandb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from wandb.sklearn import plot_feature_importances


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the data. It joins the stores data with the train data.
    It also adds new features like Year and Month and fills NaN values with 0.
    """

    data["tourney_date"] = pd.to_datetime(data["tourney_date"], format="%Y-%m-%d")

    final_data = pd.DataFrame()
    final_data["tourny_id"] = data["tourney_id"]
    final_data["tourney_name"] = data["tourney_name"]
    final_data["tourney_date"] = data["tourney_date"]
    final_data["surface"] = data["surface"]
    final_data["draw_size"] = data["draw_size"]
    final_data["tourney_level"] = data["tourney_level"]
    final_data["best_of"] = data["best_of"]
    final_data["match_num"] = data["match_num"]
    final_data["year"] = data["tourney_date"].dt.year
    final_data["month"] = data["tourney_date"].dt.month

    generator = np.random.default_rng(42)

    random_numbers = generator.random(len(data))
    # Randomly assign a player to player 1 or player 2.
    final_data["player_1_id"] = np.where(random_numbers > 0.5, data["winner_id"], data["loser_id"])
    final_data["player_2_id"] = np.where(random_numbers > 0.5, data["loser_id"], data["winner_id"])

    final_data["player_1_age"] = np.where(random_numbers > 0.5, data["winner_age"], data["loser_age"])
    final_data["player_2_age"] = np.where(random_numbers > 0.5, data["loser_age"], data["winner_age"])

    final_data["player_1_rank"] = np.where(random_numbers > 0.5, data["winner_rank"], data["loser_rank"])
    final_data["player_2_rank"] = np.where(random_numbers > 0.5, data["loser_rank"], data["winner_rank"])

    final_data["player_1_hand"] = np.where(random_numbers > 0.5, data["winner_hand"], data["loser_hand"])
    final_data["player_2_hand"] = np.where(random_numbers > 0.5, data["loser_hand"], data["winner_hand"])

    final_data["player_1_ht"] = np.where(random_numbers > 0.5, data["winner_ht"], data["loser_ht"])
    final_data["player_2_ht"] = np.where(random_numbers > 0.5, data["loser_ht"], data["winner_ht"])

    final_data["player_1_ioc"] = np.where(random_numbers > 0.5, data["winner_ioc"], data["loser_ioc"])
    final_data["player_2_ioc"] = np.where(random_numbers > 0.5, data["loser_ioc"], data["winner_ioc"])

    final_data["player_1_rank_points"] = np.where(
        random_numbers > 0.5, data["winner_rank_points"],
        data["loser_rank_points"])
    final_data["player_2_rank_points"] = np.where(
        random_numbers > 0.5, data["loser_rank_points"],
        data["winner_rank_points"])

    final_data["player_1_seed"] = np.where(random_numbers > 0.5, data["winner_seed"], data["loser_seed"])
    final_data["player_2_seed"] = np.where(random_numbers > 0.5, data["loser_seed"], data["winner_seed"])

    final_data["player_1_entry"] = np.where(random_numbers > 0.5, data["winner_entry"], data["loser_entry"])
    final_data["player_2_entry"] = np.where(random_numbers > 0.5, data["loser_entry"], data["winner_entry"])

    final_data["player_1_win_rate"] = np.where(random_numbers > 0.5, data["winner_win_rate"], data["loser_win_rate"])
    final_data["player_2_win_rate"] = np.where(random_numbers > 0.5, data["loser_win_rate"], data["winner_win_rate"])

    final_data["player_1_loss_rate"] = np.where(random_numbers > 0.5, data["winner_loss_rate"], data["loser_loss_rate"])
    final_data["player_2_loss_rate"] = np.where(random_numbers > 0.5, data["loser_loss_rate"], data["winner_loss_rate"])

    final_data["player_1_total_games_played"] = np.where(random_numbers > 0.5, data["winner_total_games_played"],
                                                         data["loser_total_games_played"])
    final_data["player_2_total_games_played"] = np.where(random_numbers > 0.5, data["loser_total_games_played"],
                                                         data["winner_total_games_played"])

    final_data["player_1_total_games_won"] = np.where(random_numbers > 0.5, data["winner_total_games_won"],
                                                      data["loser_total_games_won"])
    final_data["player_2_total_games_won"] = np.where(random_numbers > 0.5, data["loser_total_games_won"],
                                                      data["winner_total_games_won"])

    final_data["player_1_total_games_lost"] = np.where(random_numbers > 0.5, data["winner_total_games_lost"],
                                                       data["loser_total_games_lost"])
    final_data["player_2_total_games_lost"] = np.where(random_numbers > 0.5, data["loser_total_games_lost"],
                                                       data["winner_total_games_lost"])

    final_data["winner"] = np.where(random_numbers > 0.5, 0, 1)

    # Fill NaN values with Median.
    final_data["player_1_age"].fillna(final_data["player_1_age"].median(), inplace=True)
    final_data["player_2_age"].fillna(final_data["player_2_age"].median(), inplace=True)
    final_data["player_1_ht"].fillna(final_data["player_1_ht"].median(), inplace=True)
    final_data["player_2_ht"].fillna(final_data["player_2_ht"].median(), inplace=True)
    final_data["player_1_rank_points"].fillna(final_data["player_1_rank_points"].median(), inplace=True)
    final_data["player_2_rank_points"].fillna(final_data["player_2_rank_points"].median(), inplace=True)

    final_data["player_1_rank"].fillna(0, inplace=True)
    final_data["player_2_rank"].fillna(0, inplace=True)

    final_data["player_1_seed"].fillna(0, inplace=True)
    final_data["player_2_seed"].fillna(0, inplace=True)

    final_data["player_1_entry"].fillna("none", inplace=True)
    final_data["player_2_entry"].fillna("none", inplace=True)

    # Calculate the winner's age difference to the loser.
    final_data["age_difference"] = (final_data["player_1_age"] - final_data["player_2_age"]).abs()

    # Calculate the winner's rank difference to the loser.
    final_data["rank_difference"] = (final_data["player_1_rank"] - final_data["player_2_rank"]).abs()

    final_data.dropna(inplace=True)
    print(f"Number of rows after dropping NaNs: {len(final_data)}")
    print(f"Number of rows before dropping NaNs: {len(data)}")
    print(f"Number of rows dropped: {len(data) - len(final_data)}")

    final_data["player_1_hand"] = final_data["player_1_hand"].apply(lambda x: x.upper())
    final_data["player_2_hand"] = final_data["player_2_hand"].apply(lambda x: x.upper())

    final_data["player_1_ioc"] = final_data["player_1_ioc"].apply(lambda x: x.upper())
    final_data["player_2_ioc"] = final_data["player_2_ioc"].apply(lambda x: x.upper())

    final_data["player_1_entry"] = final_data["player_1_entry"].apply(lambda x: x.lower())
    final_data["player_2_entry"] = final_data["player_2_entry"].apply(lambda x: x.lower())

    return final_data


def get_train_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Load Ranking & Player Data.
    # rank_data = pd.read_csv("../data/atp_rankings_till_2022.csv")
    # player_data = pd.read_csv("../data/atp_players_till_2022.csv")

    # Process the data, join the Rank & Player data with the train data.
    train_data = process_data(train_data)
    test_data = process_data(test_data)

    # Split the data into features and target and drop columns that are meant to be predicted.
    train_features = train_data.drop(columns=["winner"])
    train_target = train_data["winner"]

    test_features = test_data.drop(columns=["winner"])
    test_target = test_data["winner"]

    # Create a ColumnTransformer to transform the data.
    column_transformer = ColumnTransformer([
        ("Drop Unused", "drop", [
            "tourny_id",
            "tourney_name",
            "player_1_id",
            "player_2_id",
            "tourney_date",
        ]),
        ("One Hot Encode", OneHotEncoder(handle_unknown="ignore"), [
            "surface",
            "tourney_level",
            "player_1_entry",
            "player_2_entry",
            "player_1_hand",
            "player_2_hand",
            "player_1_ioc",
            "player_2_ioc",
        ]),
        ("Scale", StandardScaler(), [
            "draw_size",
            "match_num",
            "player_1_age",
            "player_2_age",
            "player_1_rank",
            "player_2_rank",
            "player_1_ht",
            "player_2_ht",
            "player_1_rank_points",
            "player_2_rank_points",
            "player_1_seed",
            "player_2_seed",
            "age_difference",
            "rank_difference",
            "player_1_win_rate",
            "player_2_win_rate",
            "player_1_loss_rate",
            "player_2_loss_rate",
            "player_1_total_games_played",
            "player_2_total_games_played",
            "player_1_total_games_won",
            "player_2_total_games_won",
            "player_1_total_games_lost",
            "player_2_total_games_lost",
            "year",
            "month",
        ])
    ], remainder="passthrough")

    # Fit the transformer on the feature dataframe.
    # Reason for this is that the transformer needs to stay consistent between training and testing.
    # We don't want to process the data differently between training and testing.
    column_transformer.fit(train_features, train_target)

    return (
        pd.DataFrame.sparse.from_spmatrix(column_transformer.transform(train_features),
                                          columns=column_transformer.get_feature_names_out()),
        train_target,
        pd.DataFrame.sparse.from_spmatrix(column_transformer.transform(test_features),
                                          columns=column_transformer.get_feature_names_out()),
        test_target
    )


def maybe_start_sweep(sweep_configuration, project, entity) -> str:
    """
    This function is to easily parallelize the sweeps.
    A sweep is a hyperparameter search, where multiple models are trained with different hyperparameters.
    Basically a grid search, but with more features in our case.
    """

    # Make sure the model_output directory exists, as we will save the models there.
    pathlib.Path("./model_output").mkdir(parents=True, exist_ok=True)

    # Check if we are the leader of the sweep (The Agent that created it).
    is_leader = False
    if pathlib.Path("./.sweep_id").exists():
        with open("./.sweep_id", "r") as f:
            sweep_id = f.read().strip()
    else:
        # We are the leader, so create the sweep.
        sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
        # Save the sweep id to a file, so we can check if we are the leader next time.
        with open("./.sweep_id", "w") as f:
            is_leader = True
            f.write(f"{entity}/{project}/{sweep_id}")

    print(f"Leader: {is_leader}")
    print(f"Sweep ID: {sweep_id}")

    return sweep_id


def train_and_evaluate(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, run):
    """This function trains and evaluates the model. Just to keep the main function clean."""

    print(f"Features: {x_train.columns}")
    print(f"Target: {y_train.name}")
    print(f"Number of features: {len(x_train.columns)}")
    print(f"Number of samples: {len(x_train)}")

    # Upload the current code to wandb.
    run.log_code("./", name=f"sweep-{run.sweep_id}-code", include_fn=lambda path: path.endswith(".py"))

    # Create a KFold cross validator.
    kfold = KFold(n_splits=run.config.n_folds, shuffle=True, random_state=42)

    # Split the data into trainings data & validation data.
    test_scores = []

    train_scores = []

    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    auc_scores = []

    # Iterate over the splits.
    for index, (train_index, val_index) in enumerate(kfold.split(x_train, y_train)):
        # Get the trainings and validation data for this split.
        x_train_real, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_real, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        print(f"Strarting training of model for split {index}")
        time = datetime.now()
        # Fit the model on the training's data.
        model.fit(x_train_real, y_train_real)

        print(f"Training time: {datetime.now() - time}")

        # Evaluate the model on the different datasets.
        train_score = model.score(x_train_real, y_train_real)
        test_score = model.score(x_test, y_test)

        # Store the scores. To calculate the mean later.
        train_scores.append(train_score)
        test_scores.append(test_score)

        test_predictions = model.predict(x_test)

        # Calculate the F1 score.
        f1_score = sklearn.metrics.f1_score(y_test, test_predictions)
        accuracy_score = sklearn.metrics.accuracy_score(y_test, test_predictions)
        precision_score = sklearn.metrics.precision_score(y_test, test_predictions)
        recall_score = sklearn.metrics.recall_score(y_test, test_predictions)
        false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y_test, test_predictions)
        roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)

        f1_scores.append(f1_score)
        accuracy_scores.append(accuracy_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        auc_scores.append(roc_auc)

        # Log the scores and the model. To WandB.
        run.log({
            "split": index,
            "train_score": train_score,
            "test_score": test_score,
            "f1_score": f1_score,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "auc_score": roc_auc,
        })

        # Plot the feature importance.
        # This is a helper function from wandb. And checks if the model has a feature_importance_ attribute.
        # If it does, it will plot the feature importance.
        plot_feature_importances(model, x_train_real.columns)

        # Save the model to a file.
        with open(f"./model_output/model-{run.id}-{index}.pkl", "wb") as f:
            pickle.dump(model, f)

            if os.stat(f"./model_output/model-{run.id}-{index}.pkl").st_size > 30 * 1024 * 1024:
                print("Model too big, skipping upload")
                continue

            # Create an artifact for the model and upload it to wandb.
            art = wandb.Artifact(f"model-{run.id}", type="model")
            art.add_file(f"./model_output/model-{run.id}-{index}.pkl", name=f"model-{index}.pkl")
            run.log_artifact(art)

    # Log the mean scores.
    run.log({
        "mean_train_score": sum(train_scores) / len(train_scores),
        "mean_test_score": sum(test_scores) / len(test_scores),
        "mean_f1_score": sum(f1_scores) / len(f1_scores),
        "mean_accuracy_score": sum(accuracy_scores) / len(accuracy_scores),
        "mean_precision_score": sum(precision_scores) / len(precision_scores),
        "mean_recall_score": sum(recall_scores) / len(recall_scores),
        "mean_auc_score": sum(auc_scores) / len(auc_scores),
    })

    # Finish the run.
    run.finish()
