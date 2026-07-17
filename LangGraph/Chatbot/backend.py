from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import START , END, StateGraph
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
load_dotenv()

llm=ChatGroq(model='llama-3.3-70b-versatile')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    
def chat_node(state:ChatState):
    # take user query from state
    messages=state['messages']
    #send to llm 
    response=llm.invoke(messages)
    return {'messages':[response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

