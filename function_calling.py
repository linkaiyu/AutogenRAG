
# c# How to use Azure OpenAI GPT-4o with Function calling
# code ref: https://techcommunity.microsoft.com/t5/startups-at-microsoft/how-to-use-azure-openai-gpt-4o-with-function-calling/ba-p/4158612
# https://github.com/denisa-ms/azure-openai-code-samples/blob/main/GPT4o/Parcel%20sorting%20with%20GPT4o%20and%20functions.ipynb
# 


# from autogen import function_utils as function_utils
import function_utils as function_utils
from typing_extensions import Annotated
import _FunctionFactory_5 as functions

functions_dict = {item["func"].__name__: item["func"] for item in functions.functions_table}

tools = []
function_map = {}

# test function meta data
#for key in functions_dict:
#    f = function_utils.get_function_schema(functions_dict[key], name=functions_dict[key].__name__, description=functions_dict[key].__desc__)
    # print(f)
#    tools.append(f)
#print(tools)

import inspect

# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True


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
    tools.append(function_utils.get_function_schema(func, name=func.__name__, description=func.__desc__))
    function_map[func.__name__] = func
    return f"registering: {func.__name__} for: '{function_description}'"


# register the register_functions function
tools.append(function_utils.get_function_schema(register_functions, name=register_functions.__name__, description=register_functions.__desc__))
function_map[register_functions.__name__] = register_functions

    
user_message = """ 
I have a health insurance account with me as the primary policy holder (name linkai yu). I would like to find out the benefits of my policy. 
Summarize it in one paragraph, and then save the summary to a file c:\\temp\\output_benefit_summary.txt. 
 """
assistant_system_message = """
    For coding tasks, only use the functions you have been provided with. 
    do not generate answer on your own. do not guess. 
    for tasks that needs to access user local resources, do not generate python code. use the functions provided to you.
    if you don't have enough information to execute the task, call the given 'register_functions' function with a brief description e.g. 'get insurance policy'.
    if you need to save content to or read content from a file, call register_functions function to register functions that can save to or read from file.
    Reply TERMINATE when the task is done.
"""

messages = [
            {"role": "system", "content": assistant_system_message },
            {"role": "user", "content": user_message }
        ]


from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import json

load_dotenv()

openai = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-05-01-preview"
)

# this uses chat completion function calling feature
def call_OpenAI_using_chat_completion(messages, tools, available_functions):
    # Step 1: send the prompt and available functions to GPT
    
    while True:
        response = openai.chat.completions.create (
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # handle function call
        # code ref: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling

        if not response_message.tool_calls:
            break
        else:
            for tool_call in response_message.tool_calls:
                print(f"Recommended Function call: {tool_call}")
                print()

                # call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                function_name = tool_call.function.name

                # verify function exists
                if function_name not in available_functions:
                    return "Function " + function_name + " does not exist"
                function_to_call = available_functions[function_name]

                # verify function has correct number of arguments
                function_args = json.loads(tool_call.function.arguments)
                if check_args(function_to_call, function_args) is False:
                    return "Invalid number of arguments for function: " + function_name
                # call the function
                function_response = function_to_call(**function_args)
                print(f"Output of function call: {function_response}")
                print()
                messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
                
    return response.choices[0].message.content

import time        

def poll_run_till_completion(
    client: AzureOpenAI,
    thread_id: str,
    run_id: str,
    available_functions: dict,
    verbose: bool,
    max_steps: int = 10,
    wait: int = 3,
) -> None:
    """
    Poll a run until it is completed or failed or exceeds a certain number of iterations (MAX_STEPS)
    with a preset wait in between polls

    @param client: OpenAI client
    @param thread_id: Thread ID
    @param run_id: Run ID
    @param assistant_id: Assistant ID
    @param verbose: Print verbose output
    @param max_steps: Maximum number of steps to poll
    @param wait: Wait time in seconds between polls

    """

    if (client is None and thread_id is None) or run_id is None:
        print("Client, Thread ID and Run ID are required.")
        return
    try:
        cnt = 0
        while cnt < max_steps:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if verbose:
                print("Poll {}: {}".format(cnt, run.status))
            cnt += 1
            if run.status == "requires_action":
                tool_responses = []
                if (
                    run.required_action.type == "submit_tool_outputs"
                    and run.required_action.submit_tool_outputs.tool_calls is not None
                ):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for call in tool_calls:
                        if call.type == "function":
                            if call.function.name not in available_functions:
                                raise Exception("Function requested by the model does not exist")
                            
                            print(f"calling function: {call.function.name} args: {call.function.arguments}")
                            function_to_call = available_functions[call.function.name]
                            tool_response = function_to_call(**json.loads(call.function.arguments))
                            
                            # add tools meta data
                            tool_response = tool_response + "".join([json.dumps(tool) for tool in tools])                            
                            
                            print(f"Output: {tool_response}")
                            tool_responses.append({"tool_call_id": call.id, "output": tool_response})
                            
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_responses )
                
            if run.status == "failed":
                print("Run failed.")
                # break
            if run.status == "completed":
                break
            time.sleep(wait)

    except Exception as e:
        print(e)

# code ref: https://github.com/Azure-Samples/azureai-samples/blob/main/scenarios/Assistants/function_calling/assistants_function_calling_with_bing_search.ipynb
#   https://dev.to/airtai/function-calling-and-code-interpretation-with-openais-assistant-api-a-quick-and-simple-tutorial-5ce5
def call_OpenAI_using_assistant_function_calling(user_message, system_messages, tools, available_functions):
    print("call_OpenAI_using_assistant_function_calling")
    

    assistant = openai.beta.assistants.create(
        name="assistant",
        instructions=system_messages,
        tools=tools,
        model="gpt-4",
    )

    # Create a new thread
    thread = openai.beta.threads.create()

    # Create a new thread message with the provided task
    thread_message = openai.beta.threads.messages.create(
        thread.id,
        role="user",
        content=user_message,
    )

    # Return the assistant ID and thread ID
    # return assistant.id, thread.id

    # Create a new run for the given thread and assistant
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Loop until the run status is either "completed" or "requires_action"
    poll_run_till_completion(openai, thread.id, run.id, available_functions, verbose=True)


if __name__ == "__main__":
    call_OpenAI_using_chat_completion(messages, tools, function_map)
    # call_OpenAI_using_assistant_function_calling(user_message, assistant_system_message, tools, function_map)
    print("done")
