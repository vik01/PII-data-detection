import argparse
import bisect
import functools
import math
import random
import time
from pathlib import Path
from typing import Any

import spacy
from spacy.cli.train import train
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

OUTPUT_DIR = Path("./output/")
CONFIG = Path("./config.cfg")
TRAINED_MODEL_DIR = Path("./trained_models/")
TEST_DATA = Path("./data/test.spacy")
RESULTS = Path("./Training-Results.md")
# Store results for all models
ALL_RESULTS = []

# ----------------------------- HYPERPARAMETER TUNING -----------------------------

# Maximum number of training sessions to run
MAX_TRAINING_SESSIONS = 50
# The rate at which to introduce randomness in the selection of the next hyperparameters
EXPLORATION_RATE = 0.2
# The factor of the total range by which to increment the hyperparameters
HYPERPARAMETER_INCREMENT_FACTOR = 0.2
# Floats are rounded to 4 decimal places to avoid floating point errors
HYPERPARAMETER_PRECISION = 4
# Hyperparameter (lower bound, upper bound)
HYPERPARAMETER_LIMITS = {
    "optimizer_learn_rate": (0.0001, 0.01),
    "training_dropout": (0.1, 0.5),
    "batch_size_start": (50, 150),
    "batch_size_stop": (500, 1500),
    "batch_size_compound": (1.001, 1.01),
    "batcher_tolerance": (0.1, 0.3),
}
# The recommended order to update the hyperparameters
HYPERPARAMETER_UPDATE_ORDER = [
    "optimizer_learn_rate",
    "training_dropout",
    "batch_size_start",
    "batch_size_stop",
    "batch_size_compound",
    "batcher_tolerance",
]

# ---------------------------------------------------------------------------------


def time_func(func):
    """Wraps a function to time its execution.
    Source: https://towardsdatascience.com/a-simple-way-to-time-code-in-python-a9a175eb0172
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs) -> tuple[str, Any]:
        """Times the execution of a function and returns the time elapsed along with the function result."""
        start = time.perf_counter()

        # Call the function
        result = func(*args, **kwargs)

        time_elapsed = time.perf_counter() - start

        # Format the time elapsed in mm:ss
        time_formatted = f"{int(time_elapsed // 60):02d}:{int(time_elapsed % 60):02d}"

        return time_formatted, result

    return time_closure


@time_func
def train_model(hyperparameters: dict[str, float | int]):
    """Trains a spaCy NER model using the specified hyperparameters.

    Hyperparameters:
        optimizer_learn_rate (float): The learning rate controls how much to update the model in response to the estimated error each time the model weights are updated. Crucial for convergence speed and quality.
        training_dropout (float): The dropout rate controls the amount of regularization during training, preventing overfitting by randomly setting a fraction of input units to 0 at each update.
        batch_size_start (int): Initial batch size for the compounding batch size scheduler.
        batch_size_stop (int): Maximum batch size for the compounding batch size scheduler.
        batch_size_compound (float): Factor for compounding batch size increase per iteration, controlling how quickly the batch size grows.
        batcher_tolerance (float): Used in batch size calculation to allow for some variability in batch size, affecting memory usage and potentially training stability.
    """
    train(
        CONFIG,
        OUTPUT_DIR,
        use_gpu=0,
        overrides={
            "training.optimizer.learn_rate": hyperparameters["optimizer_learn_rate"],
            "training.dropout": hyperparameters["training_dropout"],
            "training.batcher.size.start": hyperparameters["batch_size_start"],
            "training.batcher.size.stop": hyperparameters["batch_size_stop"],
            "training.batcher.size.compound": hyperparameters["batch_size_compound"],
            "training.batcher.tolerance": hyperparameters["batcher_tolerance"],
        },
    )


def update_hyperparameters(
    best_hyperparameters: dict[str, float | int]
) -> dict[str, float | int]:
    """Updates the hyperparameters for the next training session, adjusting based on the best-performing set.

    Args:
        best_hyperparameters (dict): The best set of hyperparameters found so far.
        exploration_rate (float): The rate at which to introduce randomness in the selection of the next hyperparameters.

    Returns:
        dict[str, float | int]: The new set of hyperparameters for the next training session.
    """
    # Clone the original hyperparameters to avoid side-effects
    new_hyperparameters = best_hyperparameters.copy()

    # Randomly decide whether to explore or increment
    if random.random() < EXPLORATION_RATE:
        print("Exploring new hyperparameters...")
        # Explore: Randomly select a hyperparameter to adjust
        key = random.choice(HYPERPARAMETER_UPDATE_ORDER)
        print(f"Exploring {key}...")

        lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]

        new_value = update_value(new_hyperparameters[key], key, is_exploration=True)

        # Ensure the new value is within the bounds
        new_value = max(min(new_value, upper_bound), lower_bound)

        new_hyperparameters[key] = new_value
        print(f"Exploring {key}: {new_hyperparameters[key]}")
    else:
        # Increment: Sequentially adjust hyperparameters based on performance
        print("Incrementing hyperparameters...")
        for key in HYPERPARAMETER_UPDATE_ORDER:
            print(f"Incrementing {key}...")
            lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]

            # Randomly increment or decrement the hyperparameter
            new_value = update_value(
                new_hyperparameters[key], key, is_exploration=False
            )
            print(f"New value: {new_value}")

            if new_value > upper_bound or new_value < lower_bound:
                print(f"New value is out of bounds: [{lower_bound}, {upper_bound}]")
                continue

            new_hyperparameters[key] = new_value
            print(f"Incrementing {key}: {new_hyperparameters[key]}")
            break

    return new_hyperparameters


def update_value(value: float | int, key: str, *, is_exploration: bool) -> float | int:
    """Updates the value of a hyperparameter based on the key."""
    print(f"Updating {key}...")
    print(f"Current value: {value}")

    lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]
    if key in ["batch_size_compound", "optimizer_learn_rate"]:
        is_logarithmic = True
        # Logarithmic range
        print("Logarithmic range")
        increment = (
            math.log10(upper_bound) - math.log10(lower_bound)
        ) * HYPERPARAMETER_INCREMENT_FACTOR
    else:
        is_logarithmic = False
        # Linear range
        print("Linear range")
        increment = (upper_bound - lower_bound) * HYPERPARAMETER_INCREMENT_FACTOR

    if is_exploration:
        increment *= 2
        # Ensure the new value is different from the current value
        is_close = True
        while is_close:
            # Randomly increment or decrement the hyperparameter by a value within the range
            if is_logarithmic:
                new_value = 10 ** (
                    math.log10(value) + random.uniform(-increment, increment)
                )
            else:
                new_value = value + random.uniform(-increment, increment)
            print(f"New value: {new_value}")
            is_close = math.isclose(new_value, value)
            print(f"New value is close to current value: {is_close}")
    else:
        # Randomly increment or decrement the hyperparameter by the calculated value
        if is_logarithmic:
            new_value = 10 ** (math.log10(value) + increment * random.choice([1, -1]))
        else:
            new_value = value + increment * random.choice([1, -1])
        print(f"New value: {new_value}")

    # If hyperparameter should be an integer,
    if isinstance(value, int):
        # Round to the nearest integer
        new_value = round(new_value)
    else:
        # Round to the specified precision
        new_value = round(new_value, HYPERPARAMETER_PRECISION)

    print(f"New value: {new_value}")
    return new_value


def has_hyperparameters_been_used(hyperparameters: dict[str, float | int]) -> bool:
    """Checks if the hyperparameters have been used in a previous training session."""
    if not ALL_RESULTS:
        return False

    return any(
        all(
            math.isclose(hyperparameters[k], v)
            for k, v in result["hyperparameters"].items()
        )
        for result in ALL_RESULTS
    )


def evaluate_model(model: Path, test_data: DocBin) -> dict[str, float]:
    # Load the trained model
    print(f"Loading model from {model}...")
    nlp = spacy.load(model)

    # Load the test docs
    test_docs = list(test_data.get_docs(nlp.vocab))

    # Initialize the scorer
    scorer = Scorer(default_lang=nlp.lang, default_pipeline=nlp.pipe_names)

    # Use the model to predict the document entities. Example is a tuple of (doc, gold) where gold is the original annotated document from the test set
    predictions = [Example(nlp(doc.text), doc) for doc in test_docs]

    # Score the predictions and filter the results to only include precision, recall, and F1
    scores = scorer.score(predictions)
    return {
        key: round(float(value), 5)
        for key, value in scores.items()
        if key in ["ents_p", "ents_r", "ents_f"]
    }


def move_model_to_trained_dir(model: Path, new_name: str) -> Path:
    return model.rename(TRAINED_MODEL_DIR / f"model-{new_name}")


def save_scores(
    model: Path,
    name: str,
    training_time: str,
    scores: dict[str, float],
    hyperparameters: dict[str, float | int],
) -> dict[str, Any]:
    """Saves the scores and hyperparameters for the model to the results file, and returns the score and hyperparameters."""
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(f"| [{name}](./{TRAINED_MODEL_DIR}/{model.name}/) ")
        f.write(f"| {training_time} ")
        f.write(f"| {scores['ents_f']:0.5f} ")
        f.write(f"| {scores['ents_p']:0.5f} ")
        f.write(f"| {scores['ents_r']:0.5f} ")
        for key in HYPERPARAMETER_UPDATE_ORDER:
            f.write(f"| {hyperparameters[key]} ")
        f.write("|\n")
    return {
        "name": name,
        "path": model.name,
        "scores": scores,
        "hyperparameters": hyperparameters,
    }


def main(session: str):
    # Default hyperparameters
    hyperparameters = update_hyperparameters(
        {
            "optimizer_learn_rate": 0.0005,
            "training_dropout": 0.1,
            "batch_size_start": 100,
            "batch_size_stop": 1000,
            "batch_size_compound": 1.001,
            "batcher_tolerance": 0.2,
        }
    )

    # Create a new results file if it does not exist
    if not RESULTS.exists():
        with RESULTS.open("w", encoding="UTF-8") as f:
            f.write("# Results from Training Sessions\n")

    # Add a new section to the results file
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(f"\n\n## Automated Results (Session ({session}))\n")
        col_headers = [
            "Model",
            "Training Time",
            "F1-Score",
            "Precision",
            "Recall",
        ] + HYPERPARAMETER_UPDATE_ORDER
        for header in col_headers:
            f.write(f"| {header} ")
        f.write(f"|\n{'| - ' * len(col_headers)}|\n")

    # Load the test data
    test_data = DocBin().from_disk(TEST_DATA)

    # Run the training and evaluation process
    for i in range(MAX_TRAINING_SESSIONS):
        model_name = f"{i:02d}"
        print(f"Training model {model_name}...")

        # Train the model with the current hyperparameters and measure the time it took
        training_time, _ = train_model(hyperparameters)
        print(f"Training model {model_name} complete in {training_time}.")

        # Evaluate the model against the test data and get the scores
        output_model_path = OUTPUT_DIR / "model-best"
        scores = evaluate_model(output_model_path, test_data)

        # Move the trained model to the trained_models directory
        model_path = move_model_to_trained_dir(
            output_model_path, f"{session}-{model_name}"
        )

        # Save the scores and hyperparameters for the model
        bisect.insort(
            ALL_RESULTS,
            save_scores(model_path, model_name, training_time, scores, hyperparameters),
            key=lambda x: x["scores"]["ents_f"],
        )

        hyperparameters = update_hyperparameters(hyperparameters)
        # If these hyperparameters have been used before, try again
        while has_hyperparameters_been_used(hyperparameters):
            hyperparameters = update_hyperparameters(hyperparameters)

    # Highlight model with best score
    best = ALL_RESULTS[-1]
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(
            f"\n*Best Model: [{best['name']}](./{TRAINED_MODEL_DIR}/{best['path']}/) -> F1: {best['scores']['ents_f']}, Precision: {best['scores']['ents_p']}, Recall: {best['scores']['ents_r']}*\n"
        )


def parse_session_arg():
    parser = argparse.ArgumentParser(
        description="Train a spaCy NER model and tune hyperparameters."
    )
    parser.add_argument(
        "session",
        type=str,
        help="The name of the training session. This will be used to identify the models trained in this session.",
    )
    args = parser.parse_args()
    return args.session


if __name__ == "__main__":
    session = parse_session_arg()
    print("Starting training...")
    main(session)
    print("Training complete.")
