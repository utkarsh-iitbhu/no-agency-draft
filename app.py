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
llm = ChatGoogleGenerativeAI(model="gemini-pro")

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
    3. Are there any competitors for the idea?
    4. What is the expected timeline to complete the project?
    5. What is the budget in mind for this project?

    User Input: {user_input}

    Provide the output in JSON format without any markdown formatting.
    Output should look like this:
    {{
      "What is the project to build?": "...",
      "What are the features to add in the project?": "...",
      "Are there any competitors for the idea?": "...",
      "What is the expected timeline to complete the project?": "...",
      "What is the budget in mind for this project?": "..."      
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
    If you feel like time and budget is not enough, you can add more details to the proposal. 
    You can add details about what should be expected number of people to hire for the alloted budget and time constraints.
    
    1. Title
    Provide a catchy and crisp title for the project that reflects its core purpose and unique value proposition.

    2. Executive Summary
    Offer a concise yet comprehensive overview of the project, including its objectives, target audience, unique selling points, and expected outcomes. (150-200 words)

    3. Technology Stack Recommendation
    Present a detailed analysis of the recommended technology stack. For each major component (frontend, backend, database, etc.), provide:
    - Specific technologies/frameworks and their versions
    - Justification for each choice, considering factors like scalability, performance, community support, and alignment with project goals
    - Potential alternatives and why they were not selected

    4. Technology Tags
    Select 5-10 most relevant tags from the provided list. Dont select the same tag twice and don't select tags that are not relevant to the project. For each tag, briefly explain its relevance to the project.
    Available Tags: {all_tags}

    6. Major Features and Sub-Features:

    Provide a comprehensive breakdown of at least 10-12 key features for the project apart from the features provided by the user including both common and advanced features relevant to the user's requirements. Present the information in a structured, tabular format as shown below:
    Also provide 4-5 sub-features for each category of main feature.

    Main Feature | Sub-Features | Description | Complexity | Frontend Time (hours) | Backend Time (hours) | User Value
    -------------|--------------|-------------|------------|----------------------|----------------------|------------
    [Feature 1]  | - Sub-feature 1 | [Brief description] | [Low/Medium/High] | [UI: X, Integration: Y] | [Easy: A, Medium: B, High: C] | [Importance to user]
                 | - Sub-feature 2 | [Brief description] | [Low/Medium/High] | [UI: X, Integration: Y] | [Easy: A, Medium: B, High: C] | [Importance to user]
                 | - Sub-feature 3 | [Brief description] | [Low/Medium/High] | [UI: X, Integration: Y] | [Easy: A, Medium: B, High: C] | [Importance to user]
    [Feature 2]  | - Sub-feature 1 | [Brief description] | [Low/Medium/High] | [UI: X, Integration: Y] | [Easy: A, Medium: B, High: C] | [Importance to user]
                 | - Sub-feature 2 | [Brief description] | [Low/Medium/High] | [UI: X, Integration: Y] | [Easy: A, Medium: B, High: C] | [Importance to user]

    Guidelines:
    1. Ensure a mix of common, necessary features and advanced, innovative features that align with the project's goals.
    2. For each main feature, provide 3-4 related sub-features that enhance its functionality.
    3. Include at least 5 complex, high-value features that demonstrate deep understanding of the project domain.
    4. For complex features, provide detailed descriptions of their implementation and value to the user.
    5. Consider scalability, performance, and user experience when suggesting features.
    6. Align features with the latest industry trends and best practices in the project's domain.
    7. For each feature, specify technologies or frameworks that would be best suited for implementation.

    Remember to tailor the features to the specific project requirements provided by the user, ensuring relevance and innovation in your suggestions.

    Ensure that each feature and sub-feature is thoroughly described, with attention to development time estimates, technical implementation details, and user value. Consider the project's scope, complexity, and potential scalability needs when detailing these features.
    
    Example: 
    Main Feature | Sub-Feature | Frontend UI (hours) | Frontend Integration (hours) | Customization (%) | Backend Easy (hours) | Backend Medium (hours) | Backend High (hours) | Category
    -------------|-------------|---------------------|------------------------------|-------------------|----------------------|------------------------|----------------------|----------
    Product catalog page | Numeric pagination | 2 | 1 | 2 | 3 | 5 | 9 | E-Commerce
    Product catalog page | Card component | 0 | 1.5 | 2 | 0 | 0 | 0 | E-Commerce
    Product catalog page | Infinite scrolling | 0 | 1 | 2 | 0 | 0 | 0 | E-Commerce
    Product detail  page | Report or like review | 0 | 0.5 | 2 | 0 | 0 | 0 | E-Commerce

    Ensure that your response maintains this tabular format and includes all the specified columns. Provide realistic time estimates and customization percentages based on the complexity of each feature and sub-feature.

    Add a conclusion at the end of the project proposal.
    Ensure that each section is addressed in detail, providing specific examples, numerical data, and justifications where applicable. The response should be comprehensive, demonstrating deep understanding of the project requirements and software development best practices.
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
        # If no JSON object is found, assume no additional questions are needed
        return {"result": "No additional questions needed."}
    
    questions_json = questions_json[start_index:end_index+1]
    questions_dict = json.loads(questions_json)
    
    # Check if any questions were generated
    if all(value is None for value in questions_dict.values()):
        return {"result": "No additional questions needed."}
    
    return questions_dict
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
        # print("Extracted : ", extracted_info)
        question_chain = create_question_chain()
        questions_dict = generate_questions(question_chain, extracted_info)
        # print("Question : ", questions_dict)

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
    # print("Extracted info: ",extracted_info) # Check the final extracted info
    proposal_chain = create_proposal_chain()
    proposal = generate_proposal(proposal_chain, extracted_info)
    return render_template('proposal.html', proposal=proposal)

if __name__ == '__main__':
    app.run(debug=True)