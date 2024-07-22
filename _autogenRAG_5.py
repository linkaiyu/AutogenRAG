from pydantic import BaseModel, Field
from typing_extensions import Annotated

import autogen
import os
from dotenv import load_dotenv  

import _FunctionFactory_5 as functions


# load llm config
load_dotenv()  

config_list = [{
    'model': os.getenv("AZURE_OPENAI_MODEL"), 
    'api_key': os.getenv("AZURE_OPENAI_API_KEY"), 
    'base_url': os.getenv("AZURE_OPENAI_ENDPOINT"), 
    'api_type': 'azure', 
    'api_version': os.getenv("AZURE_OPENAI_API_VERSION"),
    'tags': ["tool", "gpt-4"]
    }]


llm_config={
    "config_list": config_list, 
    "timeout": 120,
    }


# functions dict for lookup used by FunctionFactory get_function(function_name)
functions_dict = {item["func"].__name__: item["func"] for item in functions.functions_table}


# in memory vector database for function lookup
import chromadb

documents = []
metadatas = []
ids = []

# populate the documents, metadatas and ids for the functions
for item in functions.functions_table:
    documents.append(item["func"].__desc__)
    metadatas.append({"name": item["func"].__name__})
    ids.append(item["id"])

# create the collection and add the documents
client=chromadb.Client()
collection = client.create_collection("functions")
collection.add(
        documents=documents, # we embed for you, or bring your own
        metadatas=metadatas, # filter on arbitrary metadata!
        ids=ids, # must be unique for each doc 
)


# function factory to get a function based on the description. the fuction will be called by the user proxy agent
from typing import Callable, Any, Dict  

def get_function(description: str) -> Callable[..., Any]:
    """
    use the description to find the function based on vector search for now, use the hard coded function map for testing
    
    args:
        description (str): the description of the function.
    
    returns:
        Callable[..., Any]: the function.
    """

    results = collection.query(
        query_texts=[description],
        n_results=1,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    ) 

    name = results["metadatas"][0][0]["name"]
    
    func = functions_dict.get(name, None)

    if func is not None:
        return func
    else:
        print(f"get_function() error: function found for: {description}")
        raise Exception(f"get_function fail to find function for: {description})")

# function to register other functions for agent to call, given the description    
@functions.desc("register the function for agent based on the given description")
def register_functions(function_description: Annotated[str, "description of the function to register."])  -> Annotated[str, "registration result"]:
    """
    register the functions based on the description
    
    args:
        function_description (str): the description of the function to register.
    
    returns:
        str: registration result
    """
    func = get_function(function_description)
    assistant.register_for_llm(name=func.__name__, description=func.__desc__)(func)
    user_proxy.register_for_execution(name=func.__name__)(func)
    return f"registering: {func.__name__} for: '{function_description}'"


# create user agent and assistant agent
    
user_system_message = """ 
    You are a helpful AI agent. when you talk with assistant, help them to make their response as accurate as possible to the user's requirement. make sure to execute the task in the correct order of the assistant's response.
 """
assistant_system_message = """
    For coding tasks, only use the functions you have been provided with. 
    do not generate answer on your own. do not guess. 
    for tasks that needs to access user local resources, do not generate python code. use the functions provided to you.
    if you don't have enough information to execute the task, call the given 'register_functions' function with a brief description e.g. 'get insurance policy'.
    if you need to save content to or read content from a file, call register_functions function to register functions that can save to or read from file.
    Reply TERMINATE when the task is done.
"""

import typing;

assistant = None
user_proxy = None

def Create_Agents( ) -> typing.Tuple[autogen.UserProxyAgent, autogen.AssistantAgent]:
    global assistant
    global user_proxy
    
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message=assistant_system_message,
        llm_config=llm_config,
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        system_message=user_system_message,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=12,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    # register the fundamental functions
    assistant.register_for_llm(name="register_functions", description=register_functions.__desc__)(register_functions)
    user_proxy.register_for_execution(name="register_functions")(register_functions)

    return user_proxy, assistant
    
    
# reset the agents to their initial state
def Reset_Agents():
    global user_proxy
    global assistant
    
    user_proxy.clear_history()
    user_proxy.function_map.clear()
    assistant.llm_config=llm_config

    # register the fundamental functions
    assistant.register_for_llm(name="register_functions", description=register_functions.__desc__)(register_functions)
    user_proxy.register_for_execution(name="register_functions")(register_functions)


# test the function factory
if __name__ == "__main__":
    
    print("functions table:")
    for item in functions.functions_table:
        print(f"function: {item['func'].__name__}, id: {item['id']}") 
        
    print()
    print("functions_dict:")
    for key, value in functions_dict.items():
        print(f"key: {key}, value: {value}")
        
    print()
    print("test get function find_careproviders") 
    func = functions_dict.get("find_careproviders")
    print(func.__name__)   

    print()
    print("test register_function")
    user_proxy, assistant = Create_Agents() # to initialize the agents
    for item in functions.functions_table:
        register_functions(item["func"].__desc__)
        print(f"function: {item['func'].__name__}, id: {item['id']}")

    print()
    print("test user_proxy.function_map")
    for item in user_proxy.function_map:
        print(f"key: {item}")
        
    
    while True:
        user_input = input("Enter your input: ")
        if user_input == "exit":
            break

        chat_result = user_proxy.initiate_chat(
            assistant, 
            message=user_input,  
            max_turns=10,
        )

        print("chat complete")
        
        Reset_Agents()
    

