
import random
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import (
    load_intents,
    prepare_vocabulary,
    preprocess_text,
    bag_of_words,
)


class Chatbot:

    # constructor function whenever we create an object of the class this function is called 
    # it is used as intitaliser to initliase different varaible with respect to the created object

    # the first parameter in any method of class is reserved for the self which is current object instance
    # the self refers to current object instance
    # it lets a method access and modify the data belonging to the instance on which method is called
    # self.variable creates an instance variable
    def __init__(self):
        # Locate the project files.
        self.project_root = Path(__file__).resolve().parent.parent

        # Load the original intent dataset.
        data_path = self.project_root / "data" / "intents.json"
        self.data = load_intents(data_path)

        # Recreate the same vocabulary and class ordering used in training.
        # _ used when you want to receive a value but dont intend to use it
        self.vocabulary, self.classes, _ = prepare_vocabulary(
            self.data
        )

        # Load the trained neural network.
        model_path = self.project_root / "models" / "chatbot_model.keras"

        # load_model() loads a previously trained neural network from a saved file so 
        # you can use it to make predictions without training it again.
        self.model = load_model(model_path)

        # Verify that the model's input and output sizes match.
        if self.model.input_shape[-1] != len(self.vocabulary): 
            # the input shape at the input layer must be like (,len(self.vocabulary))
            raise ValueError("Vocabulary size does not match the model.")

        if self.model.output_shape[-1] != len(self.classes):
            # the output shape be like (,len(self.classes))
            raise ValueError("Number of classes does not match the model.")

    def predict_intent(self, message):
        # Convert the message into normalized tokens.
        tokens = preprocess_text(message)

        # if empty message or just punctuation( !!! ):- no tokens
        if not tokens:
            return None, 0.0 # predicted_intent, confidence

        # Convert tokens into the numerical input expected by the model.
        vector = bag_of_words(tokens, self.vocabulary)

        # If no words match the vocabulary, avoid making a meaningless prediction.
        if sum(vector) == 0:
            # whole vector has 0 as no matches to word in the vocab
            return None, 0.0

        # Add a batch dimension: shape changes from (23,) to (1, 23).
        # sending the vector wrapped in a array
        input_data = np.array([vector], dtype=np.float32)

        # Get probabilities for all intent classes.
        # the model give result as [[0.80, 0.05, 0.10, 0.05]] for the vector
        probabilities = self.model.predict(input_data, verbose=0)[0]
        # probabilities = [0.80, 0.05, 0.10, 0.05]
        
        # Find the class with the highest probability.
        predicted_index = int(np.argmax(probabilities))

        predicted_intent = self.classes[predicted_index]
        # The model predicts the intent, not the actual sentence response. 
        # Your Python code then looks up the corresponding responses to the predicted_intent in the JSON file and randomly selects one.

        confidence = float(probabilities[predicted_index])

        return predicted_intent, confidence

    def get_response(self, message):
        tag, confidence = self.predict_intent(message)

        if tag is None:
            return "I'm not sure I understood that. Could you rephrase?"

        # Find the matching intent in the JSON dataset.
        # and returning a corresponding random response for the tag
        for intent in self.data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                return response

        return "Sorry, I couldn't find a response for that intent."



if __name__ == "__main__":
    chatbot = Chatbot()

    print("Chatbot is ready! Type 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()

        if message.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break

        response = chatbot.get_response(message)
        print("Bot:", response)