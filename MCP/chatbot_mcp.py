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

llm=ChatGroq(model='openai/gpt-oss-120b')

SERVERS={
    "math":{
        "transport":"stdio",
        "command":"C:/Users/Lenovo/AppData/Roaming/Python/Python314/Scripts/uv.exe",
        "args":[
            "run",
            "fastmcp",
            "run",
            "/Users/Lenovo/Desktop/MCP-Multi-Context-Protocol-/Math_MCP_Server/main.py"
        ]
    },
    'expense':{
        "transport":"streamable_http",
        "url":"https://test-antardhwani.fastmcp.app/mcp"
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

    result=await chatbot.ainvoke({"messages":[HumanMessage(content="Add an expense of Rs. 200 for construction material on 29 July 2026")]})

    print(result['messages'][-1].content)


if __name__=='__main__':
    asyncio.run(main())