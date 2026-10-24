import random
from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

from src.preprocessing import (
    load_intents,
    prepare_vocabulary,
    bag_of_words,
)

# load_intents: load intent data from a json file
# prepare_vocabulary: return the vocabulary,classes,
# documents(an array of tuples where each tuple contains the normalized tokens for a pattern and its corresponding intent tag)


def create_training_data(documents, vocabulary, classes):
    """Convert documents into numerical training arrays."""

    shuffled_documents = documents.copy()
    random.shuffle(shuffled_documents)

    # list of input features(itself a list) and target labels(itself a list) for training data
    train_x = []
    train_y = []

    for tokens, tag in shuffled_documents:

        # Convert the pattern's tokens into a bag-of-words vector.
        # feature is a vector array of 0s and 1s where each index corresponds to a word in the vocabulary,
        # and the value is 1 if the word is present in the tokens, otherwise 0
        features = bag_of_words(tokens, vocabulary)
        train_x.append(features)

        # Convert the intent tag into a one-hot encoded label.
        # having a unique label for each intent tag
        label = [0] * len(classes)
        label[classes.index(tag)] = 1

        train_y.append(label)

    # a tuple of two numpy arrays, one for the input features and one for the target labels
    return (
        np.array(train_x, dtype=np.float32),
        np.array(train_y, dtype=np.float32),
    )


# Machine learning models (especially neural networks) and GPUs natively prefer float32.
# It uses half the memory of Python's default 64-bit floats (float64),
# speeds up calculations significantly, and prevents type mismatch errors when feeding data into a deep learning model.


# vobabulary: a set of unique words from the training data
# classes: a set of unique intent tags from the training data
# features: a binary vector representation of the presence(1) or absence(0) of words from the vocabulary in the input tokens


def build_model(vocabulary_size, number_of_classes):
    """Build a neural network for intent classification."""

    # just defining the architecture of the neural network model, not training it yet
    # sequential model is This tells Keras that your neural network is a straight-line stack of layers
    # where data flows sequentially from the input layer all the way down to the output layer.
    model = Sequential(
        [
            # input layer with the shape of the vocabulary size, which is the number of unique words in the training data
            # expect an input vector consisiting of 0s and 1s
            Input(shape=(vocabulary_size,)),
            # dense: fully connected
            # each feature in the input vector is connected to every neuron in the layer
            # every incoming vector goes to each neuron in the layer
            # each neuron has a list of weights and a bias term that are learned during training
            # the initial value of weight and bias of each neuron is initialized by a random scheme so it make terrible guesses intially,
            # but as the model trains, it will adjust these weights and biases to make better predictions.
            # when a vector arrives at a neuron,
            # The neuron computes the dot product of the input vector and its weight vector, adds a bias, and applies the activation function such as relu to produce an output value.
            # for a single neuron, z = w1*x1 + w2*x2 + ... + wn*xn + b
            # Relu(x) = max(0, x)
            # relu: rectified linear unit, If the total number is positive, it passes it forward; if it's negative, it clips it to zero.
            # 23 vocabulary size, 128 neurons in the first hidden layer no of weigths = 23 * 128 and 128 biases for each neuron in the first hidden layer
            # total trainable parameters = (23 * 128) + 128 = 3072
            # no of parameters = (input_features * neurons_in_layer) + neurons_in_layer
            Dense(128, activation="relu"),
            Dropout(0.5),
            # Dropout is a regularization technique that randomly deactivates a fraction of neurons during training to help prevent overfitting.
            # During training, approximately half of the hidden layer's activations are dropped for each training step.
            # A different random mask is generally used on the next step.
            # This helps reduce excessive dependence on particular neurons and can reduce overfitting.
            # Without dropout
            # The neural network may rely too heavily on a small group of neurons that happen to be useful for the training examples.
            # with dropout
            # Randomly some neurons are switched off during training, encouraging the network to learn useful patterns
            # that don't depend excessively on particular neurons.
            # with dropout, the model is forced to learn more robust features that generalize better to unseen data.
            # Randomly zeros approximately half the layer's activations
            Dense(64, activation="relu"),
            Dropout(0.5),
            # output layer
            Dense(number_of_classes, activation="softmax"),
            # Each output neuron produces a value representing how strongly the model favors its corresponding class.
            # final layer's four outputs map directly to the four classes.
            # softmax activation function is used in the output layer to convert the raw output scores (logits) into probabilities that sum to 1.
        ]
    )

    # each layer passes an output vector to the next layer,
    # and the final layer produces a probability distribution over the intent classes.
    # vocabulary = hello,bye,weather and user types hello
    # the input vector becomes x = [1,0,0] (logit vector) and there are 3 classes greeting, goodbye, thanks and weather
    # let first layer has 2 neurons then the output of first layer will be a vector of 2 elements
    # no of weights in first layer each neuron has is equal to the no of features in the input vector, and each neuron has its own bias term.
    # layer 1 neuron 1, Weights: [0.8, −0.4, 0.2] · Bias: 0.1 then z1 = 1*0.8 + 0*−0.4 + 0*0.2 + 0.1 = 0.9, output1 = relu(0.9) = 0.9
    # layer 1 neuron 2, Weights: [−0.5, 0.9, 0.3] · Bias: −0.2 then z2 = 1*−0.5 + 0*0.9 + 0*0.3 + −0.2 = −0.7, output2 = relu(−0.7) = 0
    # so output vector from layer 1 is [0.9, 0] (logit vector) which is input to layer 2

    # in subsequent layers the no of weights for each neuron is equal to the number of neurons in the previous layer, and each neuron has its own bias term.

    # suppose layer 2 is final and has 4 neurons then the output of layer 2 will be a vector of 4 elements
    # every output neuron receives the output vector from previous layer. It multiplies each value by its own corresponding weight and adds a bias.
    # greeting neuron, Weights: [2.0, 0.5] · Bias: 0.0 then z1 = 0.9*2.0 + 0*0.5 + 0.0 = 1.8
    # goodbye neuron, Weights: [−1.0, 2.0] · Bias: 0.2 then z2 = 0.9*−1.0 + 0*2.0 + 0.2 = −0.7
    # thanks neuron, Weights: [0.5, 1.0] · Bias: 0.0 then z3 = 0.9*0.5 + 0*1.0 + 0.0 = 0.45
    # weather neuron, Weights: [-0.5, 2.0] · Bias: 0.1 then z4 = 0.9*−0.5 + 0*2.0 + 0.1 = −0.35

    # Softmax exponentials
    # e^1.8  = 6.050
    # e^-0.7 = 0.497
    # e^0.45 = 1.568
    # e^-0.35 = 0.705

    # Sum = 8.820

    # Probabilities
    # Greeting = 6.050 / 8.820 = 0.686
    # Goodbye  = 0.497 / 8.820 = 0.056
    # Thanks   = 1.568 / 8.820 = 0.178
    # Weather  = 0.705 / 8.820 = 0.080
    # final output vector from layer 2 is [0.686, 0.056, 0.178, 0.080] (probability vector) which is the final output of the model

    # this all is done during the forward pass of the model, and the model's predictions are compared to the true labels to compute the loss.
    # even though we considered all neuron to be on but during training, dropout randomly deactivates a fraction of neurons in each layer, 
    # which means that some of the weights and biases are not used in the forward pass for that training step.
    # this forward pass is common in both training and inference, but during inference, dropout is typically disabled, and all neurons are used to make predictions.


    # model.compile() configures the training process(configures the optimizer, loss function, and metrics.). It doesn't actually train the network or update its weights
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # categorical_crossentropy measures how far the predicted probability distribution is from the correct training labels. 
    # To calculate how different predicted probabilities are from the correct one-hot labels
    # Lower loss generally means better predictions on those examples.

    # Gradients are partial derivatives of the loss with respect to trainable parameters, such as weights and biases. 
    # They tell us how the loss changes when a parameter changes.
    # gradient is caluclated for each weight and bias in the model, and it indicates the direction and magnitude of change needed to reduce the loss.
    
    # adam optimizer uses the gradients to update the weights and biases in a way that minimizes the loss function.

    # Backpropagation calculates the gradients by propagating derivatives backward through the network using the chain rule.

    # The optimizer then uses those gradients to update the weights and biases. The entire process is often informally called the training loop

    return model
    # this gives us the model before training a configuration that tell how training will work


# model parameters: sum of parameters of all the layers in the model, including weights and biases.


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "intents.json"

    data = load_intents(data_path)
    # from the test data in intents.json file, we are extracting the patterns and their corresponding intent tags to create a
    # vocabulary of unique words, a list of unique intent classes, and a list of documents containing tokenized patterns and their associated intent tags.
    vocabulary, classes, documents = prepare_vocabulary(data)

    train_x, train_y = create_training_data(documents, vocabulary, classes)

    print("Vocabulary size:", len(vocabulary))
    print("Classes:", classes)
    print("No of Doucments:", len(documents))
    for doc in documents:
        print(doc)
    print("Training examples:", len(documents))

    for t in train_x:
        print("Input vector:", t)

    for t in train_y:
        print("Target label:", t)

    # these 14 vectors are the input features for the training data, and each vector corresponds to a pattern in the intents dataset.
    # each vector is a binary representation of the presence or absence of words from the vocabulary in the corresponding pattern.
    print("\ntrain_x shape:", train_x.shape)

    # these 14 vectors are the target labels for the training data, and each vector corresponds to a pattern in the intents dataset.
    # and also correspond to a input vector in train_x
    # each vector is a one-hot encoded unique representation of the intent tag associated with the corresponding pattern
    print("train_y shape:", train_y.shape)

    print("\nFirst input vector:")
    print(train_x[0])

    print("\nFirst target label:")
    print(train_y[0])

    print("\nTraining data prepared successfully!")

    # Defining the architecture of the neural network model, not training it yet
    model = build_model(
        vocabulary_size=len(vocabulary),
        number_of_classes=len(classes),
    )

    # displays the information about the models:- layers, dropouts, shapes, parameters, total params, trainable params
    model.summary()

    # actually trains the network by repeatedly making predictions, measuring errors, calculating gradients, and updating weights and biases.
    history = model.fit(
        train_x, # input training examples 
        train_y, # target intent labels
        epochs=200, # no of passes : One epoch means the model has processed all training examples once
        batch_size=5, # The model processes up to five examples before calculating the batch's gradients and applying an optimizer update.
        verbose=1, # for showing training progress
    )
    # returns a Keras History object. It contains the training metrics recorded across epochs.
    # verbose controls how much progress information Keras displays while running a task.
    # 0 - no progress silent, 1- a progress bar, 2 - display one summary line per epoch during training

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model.save(models_dir / "chatbot_model.keras")

    print("Model trained and saved successfully!")
