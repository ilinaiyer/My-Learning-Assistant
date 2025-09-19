from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv("/Users/ilinaiyer/My-Learning-Assistant/sample.env")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a 1 on 1 tutor for speech therapy, your goal is to provide reading comprehension questions and feedback.
        Start the conversation by asking if the user is ready to begin exercises
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chat_history = []
print("Welcome to speech therapy, type 'exit' to quit")

initial_query = "Let's start reading comprehension exercises."
raw_response = agent_executor.invoke({"query": initial_query, "chat_history": chat_history})
bot_output = raw_response["output"]
print("Bot:", bot_output)

chat_history.append({"role": "user", "content": initial_query})
chat_history.append({"role": "assistant", "content": bot_output})

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    raw_response = agent_executor.invoke({"query": query, "chat_history": chat_history})
    bot_output = raw_response["output"]
    print("Bot:", bot_output)

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": bot_output})

    


