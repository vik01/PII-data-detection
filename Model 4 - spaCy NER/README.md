# spaCy NER Model

## Setup Environment

- Install PyTorch CUDA.
- Install spaCy for GPU training models.
- Install scikit-learn.

## Training

### Manual

1. Install PyTorch and spaCy with CUDA.
2. Run [`data_pre_processing.py`](data_pre_processing.py) to generate training, validation and test data from the raw_data.
3. Edit [`config.cfg`](config.cfg) to set the correct GPU ID, and Hyperparameters.
4. Open a terminal in the folder containing `config.cfg`.
5. Train the model with `python -m spacy train config.cfg --output ./output --gpu-id 0`.

### Automated

1. Open [`automated_training.py`](./automated_training.py).
2. Adjust the hyperparameter tuning settings at the top.
3. Navigate to the `Model 4 - spaCy NER` folder in the Terminal: `cd "Model 4 - spaCy NER"`.
4. Run the file: `python automated_training.py`

## Models

You can see all these trained models in the [`trained_models`](./trained_models/) folder, named with the format `model-[total num of documents]-[train_valid_test split]`. This folder contains the training config with the hyperparameters used.

## Evaluation

The models are evaluated by their F1-Score, Precision and Recall when predicting the test same set. Multiple training sessions were performed, each with different samples of training and test data.

The following table shows the best performing models from each training session:

| Model                                                                               | Training Time * | F1-Score    | Precision   | Recall      | optimizer_learn_rate | training_dropout | batch_size_start | batch_size_stop | batch_size_compound | batcher_tolerance |
| ----------------------------------------------------------------------------------- | --------------- | ----------- | ----------- | ----------- | -------------------- | ---------------- | ---------------- | --------------- | ------------------- | ----------------- |
| [Manual Session - Model A](./Training-Results.md#manual)                            | 14 min          | **0.75482** | 0.82036     | **0.69898** | 0.001                | 0.1              | 100              | 1000            | 1.001               | 0.2               |
| [Manual Session - Model D](./Training-Results.md#manual/)                           | 13 min          | **0.75000** | 0.82317     | **0.68878** | 0.001                | 0.1              | 100              | 1000            | 1.001               | 0.2               |
| [Automated Session B - Model 06](./Training-Results.md#automated-results-session-b) | 11:48 min       | **0.74221** | 0.83439     | **0.66837** | 0.0005               | 0.18             | 100              | 1000            | 1.001               | 0.2               |
| [Automated Session B - Model 01](./Training-Results.md#automated-results-session-b) | 12:09 min       | **0.73224** | 0.78824     | **0.68367** | 0.0005               | 0.1              | 120              | 1000            | 1.001               | 0.2               |
| [Automated Session C - Model 00](./Training-Results.md#automated-results-session-c) | 17:39 min       | **0.73156** | **0.86713** | 0.63265     | 0.0002               | 0.1              | 100              | 1000            | 1.001               | 0.2               |
| [Automated Session C - Model 01](./Training-Results.md#automated-results-session-c) | 13:53 min       | 0.72892     | **0.88971** | 0.61735     | 0.0005               | 0.1              | 100              | 1000            | 1.001               | 0.2               |
| [Automated Session B - Model 05](./Training-Results.md#automated-results-session-b) | 10:36 min       | 0.71471     | **0.86861** | 0.60714     | 0.0005               | 0.1              | 140              | 1000            | 1.001               | 0.2               |
| [Automated Session A - Model 03](./Training-Results.md#automated-results-session-a) | 14:42 min       | 0.71341     | **0.88636** | 0.59694     | 0.001                | 0.34             | 100              | 1000            | 1.001               | 0.2               |


_\* Approximate time taken to train on a mobile RTX 3070Ti_

All the training results can be seen [here](./Training-Results.md).