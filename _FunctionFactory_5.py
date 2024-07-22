import autogen
import PyPDF2 
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import os
from dotenv import load_dotenv  


# wrapper function to add description to the function
# syntax ref: https://stackoverflow.com/questions/47056059/best-way-to-add-attributes-to-a-python-function
def desc(desc):
    def wrapper(f):
        f.__desc__ = desc
        return f
    return wrapper

# define custom functions:


@desc("read the content of a file")
def read_file(file_path: Annotated[str, "Name and path of file to read."]) -> Annotated[str, "file content"]:
    """
    args:
        file_path (str): the path to the file.

    returns:
        str: the content of the file.
        
    raises:
        FileNotFoundError: if the file does not exist.
    """
    print(f"read_file({file_path})")
    
    if file_path.endswith(".pdf"):    
        pdf_file = open(file_path, "rb")
        read_pdf = PyPDF2.PdfReader(pdf_file)
        number_of_pages = len(read_pdf.pages)
        
        #Give page number of the pdf file (How many page in pdf file).
        # @param Page_Nuber_of_the_PDF_file: Give page number here i.e 1

        text = ""

        for i in range(0, number_of_pages): 
            page = read_pdf.pages[i]
            text += page.extract_text()
            if len(text) > 300:
                break
            
        return text
    else:
        with open(file_path, "r") as f:
            return f.read()

@desc("save the content to a file")
def save_to_file(file_path: Annotated[str, "full path to the file"], content: Annotated[str, "content"]) -> Annotated[str, "status: success or error"]:
    """
    args:
        file_path (str): the path to the file.

    returns:
        none.
        
    raises:
        none.
    """
    
    print(f"writing file to {file_path}")
    
    with open(file_path, "w") as f:
        f.write(content)
        
    return "success"


# for now we expect the model to return this array of function names
# ["get health insurance account", "identify primary policy holder", "get policy benefits", "summarize policy benefits", "save summary to file"]
@desc("get the account number of the health insurance account of the user.")
def get_health_insurance_account(user: Annotated[str,"user name"]) -> Annotated[str,"account number"]:
    """
    Args:
        user (str): user name

    Returns:
        str: account number, e.g. A12345
    """
    print(f"get_health_insurance_account( {user})")
    return "A12345"

@desc("get the policy number of the health insurance account of the user.")
def get_health_insurance_policy(account: Annotated[str,"account number"]) -> Annotated[str,"policy name"]:
    """
    Args:
        account (str): account number e.g. A12345

    Returns:
        str: policy number, e.g. P56789
    """
    print(f"get_health_insurance_policy({account})")
    return "P56789"

@desc("get the policy benefits of a user.")
def get_policy_benefits(policy: Annotated[str,"policy number"]) -> Annotated[str,"benefits details"]:
    """
    Args:
        policy (str): policy number e.g. P56789

    Returns:
        str: the details of the policy benefits
    """
    print(f"get_policy_benefits({policy})")
    return read_file('documents/Northwind_Standard_Benefits_Details.pdf')

@desc("summarize the policy content.")
def summarize_policy_content(content_file_path: Annotated[str,"path to policy content file"]) -> Annotated[str,"summary of the policy content"]:
    """
    Args:
        content_file_path (str): document to summarize e.g. the policy benefits

    Returns:
        str: summary of the document
    """
    print(f"summarize({content_file_path})")
    # for now just hard code the summary
    summary = """ policy summary:
Inpatient Hospitalization: Covers hospital stays, including room charges, nursing care, and all related medical services.
Outpatient Services: Includes doctor visits, specialist consultations, and outpatient procedures.
Emergency Services: Covers emergency room visits and ambulance services.
Prescription Drugs: Provides coverage for prescribed medications, including both generic and brand-name drugs.
Preventive Care: Covers routine check-ups, screenings, immunizations, and other preventive services at no extra cost.
Maternity and Newborn Care: Includes prenatal care, labor, delivery, and care for the newborn baby.
Mental Health and Substance Use Disorder Services: Offers coverage for mental health services and treatments for substance use disorders.
Rehabilitation Services: Covers physical therapy, occupational therapy, and other rehabilitation services.
Laboratory Services: Provides coverage for diagnostic tests, blood work, and other laboratory services.
Pediatric Services: Includes healthcare services for children, such as dental and vision care.
"""
    return summary

@desc("find care providers near a location.") 
def find_careproviders(provider_type: Annotated[str, "type of care provider"], location: Annotated[str, "location"]) -> Annotated[str, "list of care providers"]:
    """
    Args:
        provider_type (str): type of care provider e.g. primary care physician
        location (str): location e.g. zip code

    Returns:
        str: list of care providers
    """
    print(f"find_careproviders({provider_type}, {location})")
    return "Dr. Smith, Dr. Jones, Dr. Brown"


@desc("analyze the sentiment of a text")
def analyze_sentiment(text: Annotated[str, "text to analyze"]) -> Annotated[str, "sentiment"]:
    """
    Args:
        text (str): text to analyze

    Returns:
        str: sentiment of the text
    """
    print(f"sentiment_analysis({text})")
    
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

    
    assistant = autogen.AssistantAgent(
        name='assistant',
        llm_config=llm_config,
        system_message="you are a helpful assistant"
    )

    reply = assistant.generate_reply(messages=[{"content": text, "role": "user"}])
    return reply


# a function to ask a question and get an answer using prompty as an experiment
from promptflow.core import Prompty
from pathlib import Path
BASE_DIR = Path(__file__).absolute().parent

@desc("ask a question and get an answer")
def ask_a_question(question: Annotated[str, "question to ask"]) -> Annotated[str, "answer to the question"]:
    """
    Args:
        question (str): question to ask
        
    Returns:
        str: answer to the question
    """

    if "OPENAI_API_KEY" not in os.environ and "AZURE_OPENAI_API_KEY" not in os.environ:
        # load environment variables from .env file
        load_dotenv()

    prompty = Prompty.load(source=BASE_DIR / "chat.prompty")
    output = prompty(question=question)
    return output

# continue to define more custom functions here



# functions table for function lookup used by function factory get-function(function_name)
# add your function to the table and make sure the id is unique
functions_table =  [
    {"id": "1","func": read_file},
    {"id": "2","func": save_to_file},
    {"id": "3","func": get_health_insurance_account},
    {"id": "4","func": get_health_insurance_policy },
    {"id": "5","func": get_policy_benefits },
    {"id": "6","func": summarize_policy_content },
    {"id": "7","func": find_careproviders },
    {"id": "8","func": analyze_sentiment },
    {"id": "9","func": ask_a_question},
]
