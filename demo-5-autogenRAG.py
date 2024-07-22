import autogen
import _autogenRAG_5 as autogenRAG

user_proxy, assistant = autogenRAG.Create_Agents()

# take user input prompt and call the assistant
while True:
    
    user_input = input("Enter your input: ")
    if user_input == "exit":
        break

    chat_result = user_proxy.initiate_chat(
        assistant, 
        message=user_input,  
        max_turns=12,
    )

    print("chat complete")
    
    autogenRAG.Reset_Agents()
    
    
    
    
