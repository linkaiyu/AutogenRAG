# AutogenRAG
autogen with RAG capabilities by dynamically invoking custom functions for data IO

this project demonstrate the use of Autogen to dynamically invoke user functions without hard code function registration. this allows developers to just implmenet the data IO functions without having to develop application code, and allows user to create prompts that has multiple and complex commands. This essentially allows user to create RAG application by creating prompts.

here is a description of the files:

demo-5-autogenRAG.py is the main application file
_autogenRAG_5.py is the engine that implements the dynamic function invokation mechanism
_FunctionFactory_5.py contains the custom data IO functions

to start, edit the .env file to set the azure openai api key and url. this app expects to use gpt-4 as Autogen has issues with function call using gtp-3.5

when the app starts, it will ask for user input.

you can type in the prompts, or you can use the prompt-xxx.txt files.

to use the prompt files, type this:
read file prompt-find-care-providers.txt, use the content as user input and execute it. when finish task, reply TERMINATE

you should see the user-proxy and assistant agent exchange communication for function look up, function registration and function invokation

for the flow of execution, please refer to AutogenRAG.pptx file
