
import json
import re
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer


# Download the language resources required by NLTK.
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


lemmatizer = WordNetLemmatizer()
# converts words to their base or dictionary form, which helps in reducing the dimensionality of the data
# and improving the model's ability to generalize across different word forms.
# "cats"-> "cat", "running"->"run", "better"->"good"

def tokenize(text):
    """Split text into individual tokens.""" #doc string vs code uses it on when you hover the function this is shown
    return nltk.word_tokenize(text)


def normalize_tokens(tokens):
    """Lowercase tokens and lemmatize words."""
    normalized = []

    for token in tokens:
        token = token.lower()

        # Ignore punctuation-only tokens.
        if re.fullmatch(r"\W+", token):
            continue

        normalized.append(lemmatizer.lemmatize(token))

    return normalized


def preprocess_text(text):
    """Convert raw text into normalized tokens."""
    tokens = tokenize(text) # convert the text into individual tokens array
    return normalize_tokens(tokens) 
    # return the normalized tokens array after converting to lowercase and lemmatizing the words


def load_intents(file_path):
    """Load intent data from a JSON file."""

    # open the file in read mode with utf-8 encoding and return file a file object 
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
        # json.load() function reads the JSON data from the file and converts it into a Python object (in this case, a dictionary). 


def prepare_vocabulary(data):
    """Build the vocabulary and labeled training examples."""
    vocabulary = set() 
    classes = set()
    documents = []

    # iterating through each intent object in the array intents of dataset data 
    for intent in data["intents"]:
        tag = intent["tag"]
        classes.add(tag)

        # in an intents iterate each pattern in patterns array
        for pattern in intent["patterns"]:
            tokens = preprocess_text(pattern) # tokens an array of normalized tokens for each pattern
            vocabulary.update(tokens) # update take list, tuple, set and add its elements to the set 

            # Each document is an object that contains tokens and their intent label.
            documents.append((tokens, tag))

    return sorted(vocabulary), sorted(classes), documents

# classes consist of tags sorted 
# vocabulary consist of all the normalized tokens sorted from all the patterns in the intents
# documents consist of a list of tuples, where each tuple contains the normalized tokens for a pattern and its corresponding intent tag

def bag_of_words(tokens, vocabulary):
    """Represent a sentence as a binary numerical vector."""
    token_set = set(tokens)

    res = []

    # for each word in the vocabulary check if it is present in the token_set of the input text 
    # if present then append 1 else 0
    # check if present or not record its postion in the sentence
    for word in vocabulary:
        if word in token_set:
            res.append(1)
        else: res.append(0)

    return res


# If this file is run directly: Python sets the built-in __name__ variable to "__main__", so the code inside below block runs.
# If imported: Python sets __name__ to the actual name of the file (e.g., "script"), so the block is completely skipped.

# during an import python reads file line by line and define all the function and classes in file to make it available for importing file
# it runs any direct statements or function calls sitting at the top level of the file such as print statements, db calls and server startups.

# It prevents your main execution logic (such as test runs, command-line inputs, or script-starting loops)
#  from accidentally firing when another file imports your module.

# purpose: 
# As a Reusable Library: You can import its functions and classes into other files without accidentally triggering test code, 
# database queries, or server starts upon import.
# As a Standalone Script: You can include execution logic, test cases, or command-line parsers directly inside the file so it can be 
# run on its own.

if __name__ == "__main__":
    # gives the project root dir path
    # __file__ is a special variable in Python that holds the path of the current file being executed.
    # Path converts the string path into a Path object that can be used in windows
    # clearing the path to make it fully resolved path 
    # navagates to two dir level up from current script file to the project root dir 
    # C:\Users\YourName\Projects\my_project points to this but not a string like this
    project_root = Path(__file__).resolve().parent.parent

    # the path to intents.json in data folder which is our sample dataset for training the chatbot
    # project_root a path object / operator allow us to append the folder path 
    data_path = project_root / "data" / "intents.json"

    # load the intents data from the JSON file as data it is an object with a single key "intents" which is an array of intent objects
    data = load_intents(data_path)

    # prepare these data for training the chatbot 
    vocabulary, classes, documents = prepare_vocabulary(data)

    # sample text to test the preprocessing and bag of words representation
    sample = "Hello, how are you?"

    tokens = preprocess_text(sample)

    # return how much token matches with the vocabulary of the training data in a binary vector form
    vector = bag_of_words(tokens, vocabulary)

    print("Input:", sample)
    print("Tokens of input text:", tokens)
    print("Our Test Data Vocabulary:", vocabulary)
    print("vocabulary size:", len(vocabulary))
    print("Our test doucments:", documents)
    print("Our documents size:", len(documents))
    print("Our test Classes:", classes)
    print("Classes size:", len(classes))
    print("Bag of words:", vector)
    print("Training examples:", len(documents))
