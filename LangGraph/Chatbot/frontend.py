import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG={'configurable': {'thread_id':'thread_1'}}
# to store message history -> st.session
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

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

    #calling chatbot model
    response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
    ai_message=response['messages'][-1].content

    st.session_state['message_history'].append({'role':'assistant' , 'content': ai_message })
    with st.chat_message('assistant'):
        st.text(ai_message[0]['text'])
