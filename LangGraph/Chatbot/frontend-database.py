import streamlit as st
from backend_database import chatbot , retrieve_threads
from langchain_core.messages import HumanMessage
import uuid



#******************************Utility Function*********************

def generate_threadid():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=generate_threadid()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id':thread_id}}).values['messages']

#***************************** Session Setup ***********************

# to store message history -> st.session
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]


if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_threadid()

if 'chat_thread'not in st.session_state:
        st.session_state['chat_thread']=retrieve_threads()

add_thread(st.session_state['thread_id'])



#**********************************Sidebar UI************************

st.sidebar.title("Chats")
if st.sidebar.button("New Chat "):
    reset_chat()    
    
st.sidebar.header('History')
for thread_id in st.session_state['chat_thread'][::-1]:
    if st.sidebar.button(str(thread_id)): 
        st.session_state['thread_id']=thread_id
        msg=load_conversation(thread_id)
        temp_msg=[]
        for message in msg :
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role ='assistant'
            temp_msg.append({'role':role, 'content':message.content})
        st.session_state['message_history']=temp_msg




#*********************************** Main UI*************************

#to display chat history
for message in st.session_state['message_history']: 
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input =st.chat_input("Type Here:")

if user_input:
    # adding message to message history 
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG={'configurable': {'thread_id':st.session_state['thread_id']}}


    with st.chat_message('assistant'):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({'messages': [HumanMessage(content=user_input)]},
               config=CONFIG,
                stream_mode='messages')
        )
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})
