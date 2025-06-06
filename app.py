import streamlit as st
from chatbot import Chatbot

st.set_page_config(page_title="🤖 AI Chatbot", page_icon="💬", layout="centered")

st.title("🤖 Chat with Hugging Face LLM")
st.markdown("Ask anything and get a response powered by a free LLM model.")

# Initialize chatbot
if "bot" not in st.session_state:
    st.session_state.bot = Chatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
with st.chat_message("ai"):
    st.markdown("Hi! I'm your AI assistant. Ask me anything!")

user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "text": user_input})

    # Get AI response
    response = st.session_state.bot.ask(user_input)
    st.chat_message("ai").markdown(response)
    st.session_state.messages.append({"role": "ai", "text": response})

# Optionally show chat history (optional)
with st.expander("🔁 Chat History"):
    for msg in st.session_state.messages:
        role = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"{role} {msg['text']}")
