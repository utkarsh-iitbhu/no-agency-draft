# Project Proposal Generator

This application generates detailed project proposals based on user input using AI-powered language models.

## Prerequisites

- Python 3.10.1
- pip (Python package installer)

## Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/project-proposal-generator.git
cd project-proposal-generator
```

2. Create a virtual environment:
```
python3.10 -m venv venv
```

3. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source venv/bin/activate
  ```

4. Install the required packages:
```
pip install -r requirements.txt
```

5. Set up environment variables:

Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Ensure your virtual environment is activated.

2. Run the Flask application:
```
python app.py
```

3. Open a web browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Usage

1. On the home page, enter your project details in the provided text area.
2. Click "Generate Proposal" to submit your input.
3. If additional information is needed, you'll be prompted to answer some questions.
4. After providing all necessary information, the application will generate a detailed project proposal.

## Customization

- To modify the available tags, edit the `tags-database.txt` file.
- To adjust the proposal generation template, modify the `template` variable in `app.py`.

## Troubleshooting

- If you encounter any issues with package installations, ensure you're using Python 3.10.1 and that your virtual environment is activated.
- For API-related issues, check that your Google API key is correctly set in the `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.