!pip install -qU crewai langchain-google-genai crewai_tools
import os

# For Google Colab, store your Gemini API key in userdata
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINIAI")

from crewai import Agent, Task, Crew, LLM



# Initialize the Gemini model
gemini_llm = LLM(model="gemini/gemini-2.0-flash")


# Define the agents with specific roles and the Gemini model
proffessor = Agent(
    role='Proffessor',
    goal='Define the parameters and topics of the test',
    backstory="You are an experienced proffesor who teaches DATA STRUCTURES AND ALGORITHMS",
    verbose=True,
    llm=gemini_llm
)

knowledge_base = Agent(
    role='Knowledge Base',
    goal='To give all the information for the test to be created',
    backstory="You are a really knowledgable person who has the knowledge of DATA STRUCTURES AND ALGORITHMS ",
    verbose=True,
    llm=gemini_llm
)
# Define the tasks for each agent
task1 = Task(
    description="to define parameters and topics to set the paper ",
    expected_output="A clear idea of the format of the paper and marking format ",
    agent=proffessor
)

task2 = Task(
    description="Based on the parameters like marking scheme question difficulty make well rounded paper",
    expected_output="Complete test paper for DATA STRUCTURES AND ALGORITHMS ",
    agent=  knowledge_base
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[proffessor, knowledge_base],
    tasks=[task1, task2],
    verbose=True
)

# Get your crew to work!
result = crew.kickoff()
print("######################")
print(result)
