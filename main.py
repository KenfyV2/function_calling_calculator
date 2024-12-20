from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator  # Use @validator for Pydantic 1.x
from fastapi.exceptions import RequestValidationError
from app.operations import add, subtract, multiply, divide  # Ensure correct import path
import uvicorn
import logging
import requests
import json
from dotenv import load_dotenv
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models import User, Base  # Ensure this import is correct
from app.settings import Settings
from faker import Faker
from passlib.context import CryptContext

# Load environment variables from .env file
load_dotenv()

# API Endpoint and API Key
API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("API_KEY")  # Ensure your .env file contains API_KEY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Define the database URL (example for PostgreSQL)
DATABASE_URL = f'postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}'

# Initialize SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Faker
fake = Faker()

# Initialize Passlib's CryptContext for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str, salt: str) -> str:
    """
    Hashes a password using bcrypt and an additional salt.
    """
    # Combine the plain password with the salt (acting as a pepper)
    salted_password = plain_password + salt
    return pwd_context.hash(salted_password)

def generate_fake_user(existing_emails: set, existing_usernames: set) -> User:
    """
    Generates a fake user with unique email and username.
    """
    while True:
        # Generate a random password
        plain_password = fake.password(length=12)
        
        user_data = {
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": fake.unique.email(),
            "username": fake.unique.user_name(),
            "password": plain_password
        }
        if user_data["email"] not in existing_emails and user_data["username"] not in existing_usernames:
            existing_emails.add(user_data["email"])
            existing_usernames.add(user_data["username"])
            return user_data

@app.post("/create_user")
async def create_user(db: Session = Depends(get_db)):
    try:
        logger.info("Creating a new user...")
        # Fetch existing emails and usernames to prevent duplicates
        existing_emails = set(email for (email,) in db.query(User.email).all())
        existing_usernames = set(username for (username,) in db.query(User.username).all())
        
        user_data = generate_fake_user(existing_emails, existing_usernames)
        hashed_password = hash_password(user_data["password"], settings.salt)
        user = User(
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            email=user_data["email"],
            username=user_data["username"],
            password=hashed_password
        )
        db.add(user)
        db.commit()
        logger.info(f"User created successfully.")
        
        # Add to history
        chat_history.append({"prompt": "User Data Table: Created user", "result": f"User created successfully"})
        
        return {"message": "User created successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    try:
        logger.info("Fetching users from the database...")
        users = db.query(User).all()
        logger.info(f"Fetched {len(users)} users.")
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Global list to store chat history
chat_history = []

# Function to call the Groq API
def call_groq_function(prompt, functions, model="llama3-8b-8192"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "functions": functions,
        "function_call": "auto",
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Check if the model called a function
        if "function_call" in data["choices"][0]["message"]:
            function_name = data["choices"][0]["message"]["function_call"]["name"]
            arguments = json.loads(data["choices"][0]["message"]["function_call"]["arguments"])
            return function_name, arguments

        return None, None

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return None, None

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    functions = [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            },
        },
        {
            "name": "subtract",
            "description": "Subtract two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            },
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            },
        },
        {
            "name": "divide",
            "description": "Divide two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            },
        }
    ]

    function_name, arguments = call_groq_function(prompt, functions)

    if function_name and arguments:
        if function_name == "add":
            result = add(arguments["a"], arguments["b"])
        elif function_name == "subtract":
            result = subtract(arguments["a"], arguments["b"])
        elif function_name == "multiply":
            result = multiply(arguments["a"], arguments["b"])
        elif function_name == "divide":
            result = divide(arguments["a"], arguments["b"])
        else:
            result = "Error: Unknown function called"
    else:
        result = "No function call was made or an error occurred."

    # Store the prompt and result in the chat history
    chat_history.append({"prompt": f"Prompt: {prompt}", "result": f"Result: {result}"})

    return {"result": result}

@app.get("/history")
async def get_history():
    """
    Retrieve the chat history.
    """
    logger.info("Fetching chat history...")
    logger.info(f"Chat history: {chat_history}")
    return {"history": chat_history}

# Pydantic model for request data
class OperationRequest(BaseModel):
    a: float = Field(..., description="The first number")
    b: float = Field(..., description="The second number")

    @field_validator('a', 'b')  # Correct decorator for Pydantic 1.x
    def validate_numbers(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError('Both a and b must be numbers.')
        return value

# Pydantic model for successful response
class OperationResponse(BaseModel):
    result: float = Field(..., description="The result of the operation")

# Pydantic model for error response
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")

# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Extracting error messages
    error_messages = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()])
    logger.error(f"ValidationError on {request.url.path}: {error_messages}")
    return JSONResponse(
        status_code=400,
        content={"error": error_messages},
    )

@app.get("/")
async def read_root(request: Request):
    """
    Serve the index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add", response_model=OperationResponse, responses={400: {"model": ErrorResponse}})
async def add_route(operation: OperationRequest):
    """
    Add two numbers.
    """
    try:
        result = add(operation.a, operation.b)
        
        # Add to history
        chat_history.append({"prompt": f"Calculator: {operation.a} + {operation.b}", "result": f"Result: {result}"})
        
        return OperationResponse(result=result)
    except Exception as e:
        logger.error(f"Add Operation Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/subtract", response_model=OperationResponse, responses={400: {"model": ErrorResponse}})
async def subtract_route(operation: OperationRequest):
    """
    Subtract two numbers.
    """
    try:
        result = subtract(operation.a, operation.b)
        
        # Add to history
        chat_history.append({"prompt": f"Calculator: {operation.a} - {operation.b}", "result": f"Result: {result}"})
        
        return OperationResponse(result=result)
    except Exception as e:
        logger.error(f"Subtract Operation Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/multiply", response_model=OperationResponse, responses={400: {"model": ErrorResponse}})
async def multiply_route(operation: OperationRequest):
    """
    Multiply two numbers.
    """
    try:
        result = multiply(operation.a, operation.b)
        
        # Add to history
        chat_history.append({"prompt": f"Calculator: {operation.a} * {operation.b}", "result": f"Result: {result}"})
        
        return OperationResponse(result=result)
    except Exception as e:
        logger.error(f"Multiply Operation Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/divide", response_model=OperationResponse, responses={400: {"model": ErrorResponse}})
async def divide_route(operation: OperationRequest):
    """
    Divide two numbers.
    """
    try:
        result = divide(operation.a, operation.b)
        
        # Add to history
        chat_history.append({"prompt": f"Calculator: {operation.a} / {operation.b}", "result": f"Result: {result}"})
        
        return OperationResponse(result=result)
    except ValueError as e:
        logger.error(f"Divide Operation Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Divide Operation Internal Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)