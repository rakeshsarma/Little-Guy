#importing dependencies
from dotenv import load_dotenv
import os
import pandas as pd
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langcorn import create_service
import spacy
from langchain.memory import ConversationEntityMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.memory import ConversationKGMemory
from langchain.memory import CombinedMemory
from lcserve import serving


load_dotenv('allkeys.env')



OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

document = pd.read_csv('KAG_conversion_data.csv')
document['Date'] = pd.to_datetime(document['Date'], format="%d-%m-%Y")


llm_code = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k") #gpt-3.5-turbo-16k-0613
llm_context = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-16k") #gpt-3.5-turbo

chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
    )
chat_history_summary = ConversationSummaryMemory(
    llm=llm_context, 
    memory_key="chat_history_summary",
    input_key="input"
    )
chat_history_KG = ConversationKGMemory(
    llm=llm_context, 
    memory_key="chat_history_KG",
    input_key="input",
    )

memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])

little_guy_with_memory = create_pandas_dataframe_agent(
    llm = llm_code, 
    df = document, 
    verbose=True, 
    agent_executor_kwargs={"memory": memory},
    input_variables=['df_head', 'input', 'agent_scratchpad', 'chat_history_buffer', 'chat_history_summary', 'chat_history_KG']
    )

@serving
def ask(question: str):
    question = question+"Use python_repl_ast."
    answer = little_guy_with_memory.run(question)
    return answer




#answer = little_guy_with_memory.run()

if __name__ == "__main__":
    ask('How many rows are there in the dataset?')
    
