from langgraph.graph import StateGraph , START ,END
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage , HumanMessage,AIMessage , BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm=ChatGroq(model='llama-3.3-70b-versatile')

SERVERS={
    "math":{
        "transport":"stdio",
        "command":"uv",
        "args":[
            "run",
            "fastmcp",
            "run",
            "/Users/Lenovo/Desktop/MCP-Multi-Context-Protocol-/Math_MCP_Server/main.py"
        ]
    }
}

client=MultiServerMCPClient(SERVERS)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]



async def build_graph():
    tools=await client.get_tools()
    print(tools)
    llm_with_tools=llm.bind_tools(tools)

    #node
    async def chat_node(state:ChatState):
        messages=state['messages']
        response=await llm_with_tools.ainvoke(messages)
        return {'messages':[response]}

    tool_node=ToolNode(tools)

    graph=StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START , "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot=graph.compile() 
    return chatbot

async def main():
    chatbot=await build_graph()

    result=await chatbot.ainvoke({"messages":[HumanMessage(content="print the addition of 8908 and 4444")]})

    print(result['messages'][-1].content)


if __name__=='__main__':
    asyncio.run(main())