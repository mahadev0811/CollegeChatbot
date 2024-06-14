import streamlit as st
import google.generativeai as genai
from FlagEmbedding import FlagModel
from numpy import dot, array
from audio_recorder_streamlit import audio_recorder
from pickle import load as pkl_load
import speech_recognition as sr
from io import BytesIO
from json import load as jsn_load


# constants
embeddings_fl, data_txt_fl = 'gat_embeddings.pkl', 'data_generation/gat_refined.txt'
max_convs = 30 # max number of messages to remember(bot+user)
genai.configure(api_key = jsn_load(open("config.json"))["google_api_key"])

# @st.cache_data
# def set_bg(bg_img):

#     ext = bg_img.split('.')[-1]

#     st.markdown(
#         f"""
#          <style>
#          .stApp {{
#              background: url(data:image/{ext};base64,{b64encode(open(bg_img, "rb").read()).decode()});
#              background-size: cover;
#          }}
#          </style>
#          """,
#         unsafe_allow_html=True
#     )

#     return

# load data
@st.cache_resource
def load_data():
    with open(data_txt_fl, 'rb') as f:
        lines = f.readlines()

    gat_data = ''.join([line.decode('utf-8').replace('\r\n', '\n') for line in lines])
    with open(embeddings_fl, 'rb') as f:
        data_embeddings = pkl_load(f)
    return gat_data.split('\n\n'), data_embeddings    

def fetch_relevant_paragraphs(query, data_embeddings, top_k=2):

    model = FlagModel('BAAI/bge-base-en-v1.5')
    query_embedding = model.encode(query)
    similarity = dot(query_embedding, array(data_embeddings).T)
    return similarity.argsort()[-top_k:][::-1]

def prompt_augmentation(query, para):

    history_prompt_template = '''
                            QUESTION:
                            {user_input}

                            INSTRUCTIONS:
                            Consider the user's QUESTION in the context of our previous conversation. 
                            Use only the information exchanged so far to formulate your response. 
                            If the necessary information is not available in our conversation history, please respond with "-1".
                            '''
    
    if para == "":
        return history_prompt_template.format(user_input=query)

    prompt_template = '''
                  DOCUMENT:
                  {context}

                  QUESTION:
                  {user_input}

                  INSTRUCTIONS:
                  Answer the users QUESTION using the DOCUMENT text above
                  Keep your answer ground in the facts of the DOCUMENT
                  Properly format the answer to the QUESTION and elaborate on the answer where necessary
                  Never mention about the DOCUMENT or the QUESTION in the answer
                  If the DOCUMENT doesn't contain the facts to answer the QUESTION, please respond with "-1"
                  '''
    
    return prompt_template.format(context=para, user_input=query).replace('\n', ' ')

def convert_audio_to_text(audio_bytes):

    recognizer = sr.Recognizer()
    audio_file = BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except Exception as e:
            return None
        
def fetch_query(audio_bytes):
    
    text = st.chat_input("Type your question here", key="text")
    if text:
        return text
    
    # audio or keyboard input
    if audio_bytes:
        text = convert_audio_to_text(audio_bytes)
        return text

st.set_page_config(page_title="Chatbot",page_icon="💬")
# set_bg("main_screen.jpg")

st.title("GAT - College Q&A Chatbot")
st.caption("Find answers to any of your questions related to Global Academy of Technology")

if "history" not in st.session_state:
    st.session_state.history = []

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history = st.session_state.history)
audio_bytes = audio_recorder(icon_size="2x", key="audio")

if 'messages' not in st.session_state:
        st.session_state.messages = [
            {'role': 'assistant', 'content': 'How may I help you today?'}
        ]

with st.sidebar:
    # add a switch to turn on "follow-up questions" mode
    follow_up_mode = st.checkbox("Follow-up Questions Mode", key="follow_up_mode")
    if follow_up_mode:
        st.info("When turned ON, the chatbot will try to answer your questions based on the conversation history and data-documents.")
    else:
        st.info("When turned OFF, the chatbot will try to answer your questions based on the data-documents only.")

    # option to set the number of paragraphs to search in
    search_depth = st.slider("Search Depth", min_value=1, max_value=3, value=2, step=1)
    st.info("As search depth increases, the probability of finding the right answer increases but the response time also increases.")

    if st.button("Clear Chat Window", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.session_state.messages = [
            {'role': 'assistant', 'content': 'How may I help you today?'}
        ]
        st.rerun()

# display the chat messages
for message in st.session_state.messages:
    role = message['role']
    with st.chat_message(role):
        msg_txt = message['content']
        if role == "assistant" and msg_txt == "-1":
            continue
        st.markdown(msg_txt)        

gat_data_lst, data_embeddings = load_data()
if len(st.session_state.messages) >= max_convs:
    st.warning("You have reached the maximum number of messages. Please clear the chat window to continue.")
    st.stop()

# fetch user query
query = fetch_query(audio_bytes)
if not query:
    st.stop()

# start the conversation only if the user has entered a query
with st.chat_message("user"):
    st.markdown(query)

st.session_state.messages.append({'role': 'user', 'content': query})

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    message_placeholder.markdown("Thinking...")

    # first try to answer it using only history (follow-up questions)
    if follow_up_mode and len(st.session_state.history) > 0:
        prompt = prompt_augmentation(query, "")
        try:
            response = chat.send_message(prompt, stream=False)
        except Exception as e:
            print(e)
            st.markdown("Something went wrong. Please try again.")
            st.stop()

        resp_lower = response.text.lower()
        if not (("-1" in resp_lower) or ("document" in resp_lower and "not" in resp_lower) or ("context" in resp_lower and "not" in resp_lower) or 
                ("conversation" in resp_lower and "not" in resp_lower) or ("sorry" in resp_lower)):
            message_placeholder.markdown(response.text)
            st.session_state.messages.append({'role': 'model', 'content': response.text})
            st.session_state.history = chat.history
            st.stop()

    chat.history = []
    top_idx = fetch_relevant_paragraphs(query, data_embeddings, search_depth)
    combined_para = "\n\n".join([gat_data_lst[idx] for idx in top_idx])
    message_placeholder.markdown(f"Searching in {[gat_data_lst[idx].splitlines()[0].split(':- ')[1].split('/')[0] for idx in top_idx]} webpages...")
    prompt = prompt_augmentation(query, combined_para)

    # without streaming
    # response = chat.send_message(prompt, stream=False)
    # resp_lower = response.text.lower()
    # if ("-1" in resp_lower) or ("document" in resp_lower and "not" in resp_lower) or ("context" in resp_lower and "not" in resp_lower) or ("sorry" in resp_lower):
    #     message_placeholder.markdown("Sorry, I couldn't find the answer to your question.\nRetry by adding more context.")
    #     st.session_state.messages.append({'role': 'model', 'content': "Sorry, I couldn't find the answer to your question.\nRetry by adding more context."})

    # else:
    #     message_placeholder.markdown(response.text)
    #     st.session_state.messages.append({'role': 'model', 'content': response.text})

    # with streaming
    try:
        response = chat.send_message(prompt, stream=True)
        full_response = ''
        for chunk in response:
            if chunk.text == None:
                continue
            full_response += chunk.text + ' '
            resp_lower = full_response.lower()
            if ("-1" in resp_lower) or ("document" in resp_lower and "not" in resp_lower) or ("context" in resp_lower and "not" in resp_lower) or ("sorry" in resp_lower):
                full_response = "Sorry, I couldn't find the answer to your question.\nRetry by adding more context."
                break
            message_placeholder.markdown(full_response + '▌')

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({'role': 'model', 'content': full_response})
        st.session_state.history = chat.history

    except Exception as e:
        print(e)
        st.warning("Something went wrong. Please try again.")
        chat.history = []           # clear the history as there can be an incomplete iterator in the chat object
        st.session_state.history = chat.history

