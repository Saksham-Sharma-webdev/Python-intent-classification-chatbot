import streamlit as st
from src.chatbot import Chatbot


# Configure the webpage
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# Display the application heading
st.title("🤖 AI Chatbot")
st.caption("Powered by Python, TensorFlow and NLP")
st.write(
    "Welcome! I'm an intent-classification chatbot. "
    "Ask me something, and I'll do my best to help."
)

# Clear the conversation when the button is clicked
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()  # Refresh the app to show the empty conversation
st.divider()


# Load the chatbot only once and reuse it
# Caching the Chatbot instance avoids repeatedly loading those resources during ordinary reruns. 
# It remains cached according to Streamlit's caching rules, rather than being recreated on every message.
@st.cache_resource
def load_chatbot():
    return Chatbot()


# Handle errors that occur while loading the chatbot
try:
    chatbot = load_chatbot()
except Exception as error:
    st.error(f"Could not load chatbot: {error}")
    st.stop()  # Stop the app if the chatbot cannot be loaded


# Store conversation history between Streamlit reruns
# Streamlit typically reruns your Python script when a user interacts with a widget. 
# Without session state, your previous messages would be lost when the script runs again.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Create the chat input box
prompt = st.chat_input("Type your message here...") 


# Process the user's message
if prompt:
    # Save and display the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # show the input prompt on the screen as markdown
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the chatbot's response
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # getting the response from the chatbot
                response = chatbot.get_response(prompt)

            st.markdown(response)

        # Save the response so it remains visible after reruns
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as error:
        st.error(f"Something went wrong: {error}")



# Step 1: User enters a message
# st.chat_input() receives "Hello" when the user submits it.

# Step 2: Save the conversation
# st.session_state.messages.append(...) adds the new message to the conversation history.

# Step 3: Generate the response
# chatbot.get_response(prompt) calls your existing chatbot logic. The spinner shows while it processes the message.

# Step 4: Display and save the reply
# The assistant's message appears on the page and is saved in session state so it can be displayed again during the next rerun.