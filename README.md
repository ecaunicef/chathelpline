# ChatHelpline

## Requirements:
- Before starting, ensure you have the following installed:
    1. Python version: 3.10
    2. Pip (Python package manager)
    3. A virtual environment tool like venv or virtualenv

## Installations:

1. Clone the repository:

```bash
git clone https://github.com/ecaunicef/mychildhelpline-app.git

cd mychildhelpline-app/

git checkout chathelpline
```

2. Create Virtual Environment:
- For Windows:

```bash
python -m venv venv
venv/Scripts/activate
```

- For Linux/MacOs:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt

```

## Running the Project

- Run the development server

```bash
python manage.py runserver
```

The Api will be available at: http://127.0.0.1:8000/


## API Endpoints:

1. Chatbot Query: `combined_intent/`


## API Testing Using Postman

- Method: `GET`
- Url: `http://127.0.0.1:8000/combined_intent/?message=Query &lang=en`

- Eg: `http://127.0.0.1:8000/combined_intent/?message=Hello I am happy&lang=en`









      
