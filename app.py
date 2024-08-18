import os
import json
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain import LLMChain
from flask import Flask, request, render_template
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-pro",max_output_tokens=18000)

def load_tags():
    """
    Loads tags from a file named 'tags-database.txt' and returns them as a comma-separated string.

    Parameters:
    None

    Returns:
    str: A comma-separated string of tags.
    """
    with open('tags-database.txt', 'r') as file:
        return ', '.join(file.read().splitlines())

all_tags = load_tags()

def create_extraction_chain():
    """
    Creates a chain for extracting specific information from user input.

    This function generates a prompt template for extracting details about a project, 
    including the project description, features, competitors, timeline, and budget. 
    It then uses this template to create an LLMChain instance, which can be used to 
    process user input and extract the desired information.

    Returns:
        LLMChain: A chain for extracting project information from user input.
    """
    template = """
    Extract the following information from the user's input. If the information is not present, output "Not provided":

    1. What is the project to build?
    2. What are the features to add in the project?
    
    User Input: {user_input}

    Provide the output in JSON format without any markdown formatting.
    Output should look like this:
    {{
      "What is the project to build?": "...",
      "What are the features to add in the project?": "...",
    }}
    """
    prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(template)])
    return LLMChain(llm=llm, prompt=prompt)

def create_question_chain():
    template = """
    Based on the extracted information, generate questions for any missing or unclear details:

    {extracted_info}

    Generate questions only for items marked as "Not provided" or if more clarification is needed.
    If all information is provided, respond with "No additional questions needed."

    Provide the output output in JSON format without any markdown formatting.
    """
    prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(template)])
    return LLMChain(llm=llm, prompt=prompt)

def create_proposal_chain():
    template = """
    Create a highly detailed and comprehensive project proposal based on the following information:
    {all_info}

    Using the provided user input and your expertise in software development and project management, create an extensive project proposal. 
    Follow the structure below, ensuring each section is thoroughly addressed with specific details, examples, and justifications where applicable.
    The features provided by the user are basic so you have to add new complex and multipurpose features which can be around the project 
        
    1. Title: It should be 2-3 words of title
    Provide a catchy and crisp title for the project that reflects its core purpose and unique value proposition.

    2. Executive Summary: It should be 2-3 lines of title
    Offer a concise yet comprehensive overview of the project, including its objectives, target audience, unique selling points, and expected outcomes.

    3. Technology Stack Recommendation
    Present a detailed analysis of the recommended technology stack. For each major component (frontend, backend, database, etc.), provide:
    - Specific technologies/frameworks and their versions for front-end you can also specify the type of CSS we can use like tailwind or so
    - Justification for each choice, considering factors like scalability, performance, community support, and alignment with project goals
    - Potential alternatives and why they were not selected

    4. Technology Tags
    Select 5-10 most relevant tags from the provided list. Dont select the same tag twice and don't select tags that are not relevant to the project. For each tag, briefly explain its relevance to the project.
    Available Tags: {all_tags}

    6. Major Features and Sub-Features:

    Provide a comprehensive breakdown of at least 10-12 key features features for the project, representing the user's journey through the application. Apart from the features provided by the user including both common and advanced features relevant to the user's requirements.
    Present the information in a structured format as shown below.
    Also provide 4-5 sub-features for each category of main feature.

    Main Feature | Sub-Features | Description | Complexity | Frontend Time (hours) | Backend Time (hours) | User Value
    -------------|--------------|-------------|------------|----------------------|----------------------|------------
    [Feature 1]  | - Sub-feature 1 | [Brief description] | [Low/Medium/High] | [UI and Integration: X] | [Backend and Integration: X] | [Importance to user][Low/Medium/High] 
                 | - Sub-feature 2 | [Brief description] | [Low/Medium/High] | [UI and Integration: X] | [Backend and Integration: X] | [Importance to user][Low/Medium/High]
                 | - Sub-feature 3 | [Brief description] | [Low/Medium/High] | [UI and Integration: X] | [Backend and Integration: X] | [Importance to user][Low/Medium/High]
    [Feature 2]  | - Sub-feature 1 | [Brief description] | [Low/Medium/High] | [UI and Integration: X] | [Backend and Integration: X] | [Importance to user][Low/Medium/High]
                 | - Sub-feature 2 | [Brief description] | [Low/Medium/High] | [UI and Integration: X] | [Backend and Integration: X] | [Importance to user][Low/Medium/High]

    Guidelines:
    1. Structure the features to reflect a typical user's journey through the application, from initial interaction to advanced usage.
    2. Begin with onboarding features (e.g., authentication, user profile setup) and progress through core functionalities to auxiliary features.
    3. For each main feature, provide 4-5 related sub-features that enhance its functionality.
    4. Include both essential features and advanced features that add unique value to the project. You can add a superscript of `recommended` if it is not provided by the user but it can fit well with the projects idea. 
    5. Ensure the features cover all aspects of the application, including user interface, backend processes, and third-party integrations where relevant.
    6. Consider the entire lifecycle of user interaction, including post-purchase or long-term engagement features.
    7. For complex features, provide more detailed descriptions of their implementation and value to the user.
    8. Align features with the latest industry trends and best practices in the project's domain.
    9. For each feature, specify technologies or frameworks that would be best suited for implementation.

    Remember to tailor the features to the specific project requirements provided by the user, ensuring relevance and innovation in your suggestions while maintaining a logical flow that represents the user's progression through the application.
    
    Ensure that each feature and sub-feature is thoroughly described, with attention to development time estimates, technical implementation details, and user value. Consider the project's scope, complexity, and potential scalability needs when detailing these features.
    
    Example: 
    
    | Main Feature          | Sub-Feature            | Description                                                                | Complexity        | Frontend Time (hours)     | Backend Time (hours)       | User Value           |
    |-----------------------|------------------------|----------------------------------------------------------------------------|-------------------|---------------------------|----------------------------|----------------------|
    | Product Catalog Page  | - Numeric pagination   | Allows users to navigate through multiple pages of products using numeric links. | Medium           | 3     | 4 | High                 |
    |                       | - Card component       | Displays products in a grid format with relevant details in each card.          | Low              | 1.5   | 2 | Medium               |
    |                       | - Infinite scrolling   | Automatically loads more products as the user scrolls down the page.           | Medium           | 2     | 3 | High                 |
    | Product Detail Page   | - Report or like review| Allows users to report inappropriate reviews or like helpful ones.              | Low              | 0.5   | 2.5 | Medium               |

    This table format provides a structured overview of features, breaking down sub-features, their descriptions, complexities, and time estimates for both frontend and backend work, along with the importance of each feature to the user.

    Timeline Breakdown:

    Development Timeline:
    - Frontend Development: [X] hours
    - Backend Development: [X] hours
    - Database Design and Implementation: [X] hours
    - API Integration: [X] hours
    - Server Setup and Configuration: [X] hours
    - Continuous Integration/Continuous Deployment Setup: [X] hours
 
    Total Estimated Timeline: [Sum of all hours(show the timeline in days/weeks/months)] INR (average employee works for 8 hours a day)

    """
    prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(template)])
    return LLMChain(llm=llm, prompt=prompt)

def extract_information(chain, user_input):
    """
    Extracts information from the provided user input using the given chain.

    Parameters:
        chain (object): The chain object used for information extraction.
        user_input (str): The user input to extract information from.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    extraction_result = chain.run(user_input=user_input)
    start_index = extraction_result.find('{')
    end_index = extraction_result.rfind('}')
    extraction_result = extraction_result[start_index:end_index+1]
    return json.loads(extraction_result)

def generate_questions(chain, extracted_info):
    """
    Generates a list of questions based on the provided extracted information.

    Parameters:
        chain (object): The chain object used for question generation.
        extracted_info (dict): The extracted information to generate questions from.

    Returns:
        dict: A dictionary of questions or a string indicating no additional questions are needed.
    """
    questions_json = chain.run(extracted_info=json.dumps(extracted_info, indent=2))
    start_index = questions_json.find('{')
    end_index = questions_json.rfind('}')
    
    if start_index == -1 or end_index == -1:
        # If no JSON object is found, generate questions for missing information
        return generate_questions_for_missing_info(extracted_info)
    
    questions_json = questions_json[start_index:end_index+1]
    questions_dict = json.loads(questions_json)
    
    # Check if any questions were generated
    if all(value is None for value in questions_dict.values()):
        return generate_questions_for_missing_info(extracted_info)
    
    return questions_dict

def generate_questions_for_missing_info(extracted_info):
    """
    Generates questions for missing information in the extracted info.

    Parameters:
        extracted_info (dict): The extracted information to check for missing data.

    Returns:
        dict: A dictionary of questions for missing information.
    """
    questions = {}
    for key, value in extracted_info.items():
        if value == "Not provided":
            questions[key] = f"Please provide information for: {key}"
    return questions if questions else {"result": "No additional questions needed."}

def generate_proposal(chain, all_info):
    """
    Generates a proposal using the given chain and information.

    Args:
        chain (object): The chain object used for generating the proposal.
        all_info (str): The information to generate the proposal from.

    Returns:
        object: The generated proposal.
    """
    return chain.run(all_info=all_info, all_tags=all_tags)

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the root route of the application.

    This function is responsible for handling the root route of the application, which is accessed via a GET or POST request. If the request method is POST, the function extracts the user input from the request form, creates an extraction chain, extracts information from the user input using the extraction chain, generates questions about the extracted information using a question chain, and checks if there are any additional questions needed. If there are no additional questions needed, it generates a proposal using a proposal chain and renders the 'proposal.html' template with the proposal. Otherwise, it extracts the question texts and renders the 'questions.html' template with the questions and the extracted information. If the request method is GET, it renders the 'index.html' template.

    Parameters:
    None

    Returns:
    - If the request method is POST and there are no additional questions needed, it renders the 'proposal.html' template with the proposal.
    - If the request method is POST and there are additional questions needed, it renders the 'questions.html' template with the questions and the extracted information.
    - If the request method is GET, it renders the 'index.html' template.
    """
    if request.method == 'POST':
        user_input = request.form['user_input']
        extraction_chain = create_extraction_chain()
        extracted_info = extract_information(extraction_chain, user_input)
        # print("Extracted : ", extracted_info) # Uncomment to find the extracted info from the user's draft
        question_chain = create_question_chain()
        questions_dict = generate_questions(question_chain, extracted_info)
        # print("Question : ", questions_dict) # Check the questionaaire created by the llm for incomplete data

        if "result" in questions_dict and questions_dict["result"] == "No additional questions needed.":
            proposal_chain = create_proposal_chain()
            proposal = generate_proposal(proposal_chain, extracted_info)
            return render_template('proposal.html', proposal=proposal)
        else:
            return render_template('questions.html', questions_dict=questions_dict, extracted_info=extracted_info)
    return render_template('index.html')

def generate_questionnaire(input_dict):
    """
    Generate a questionnaire based on the provided input dictionary.

    Args:
        input_dict (dict): A dictionary containing the user input and extracted information.

    Returns:
        tuple: A tuple containing two elements:
            - questionnaire (list): A list of questions and their corresponding answers.
            - extracted_info (dict): The updated extracted information dictionary.

    This function parses the 'extracted_info' string from the input dictionary into a dictionary. It then iterates over each key-value pair in the extracted_info dictionary. If the value is "Not provided", it checks if there is a corresponding answer in the main input_dict. If there is, it substitutes the "Not provided" with the answer and appends the question and answer to the questionnaire list. If there is no answer, it keeps it as a question and appends it to the questionnaire list. Finally, it returns the questionnaire list and the updated extracted_info dictionary.
    """
    extracted_info = json.loads(input_dict['extracted_info'].replace("'", '"'))
    questionnaire = []
    for key, value in extracted_info.items():
        if value == "Not provided":
            if key in input_dict:
                extracted_info[key] = input_dict[key]
                questionnaire.append(f"Q: {key}\nA: {input_dict[key]}")
            else:
                questionnaire.append(f"Q: {key}\nA: ")
    return questionnaire, extracted_info

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    """
    Handles the submission of answers from the user.

    This function is triggered when the user submits their answers to the questionnaire.
    It extracts the answers from the form, generates a proposal based on the answers,
    and returns the proposal as an HTML template.

    Parameters:
        None

    Returns:
        render_template: The proposal HTML template with the generated proposal.
    """
    answers = request.form.to_dict()
    _ , extracted_info = generate_questionnaire(answers)
    # print("info: ",extracted_info) # Check the final extracted info
    proposal_chain = create_proposal_chain()
    proposal = generate_proposal(proposal_chain, extracted_info)
    return render_template('proposal.html', proposal=proposal)

if __name__ == '__main__':
    app.run(debug=True)