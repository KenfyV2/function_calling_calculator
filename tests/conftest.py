import os
import subprocess
import time
from typing import Generator

import pytest
import requests
from dotenv import load_dotenv
from faker import Faker
from passlib.context import CryptContext
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.models import Base, User  # Ensure these imports are correct based on your project structure
from app.schemas import UserData
from app.settings import Settings  # Adjust the import path based on where Settings is defined


# Initialize Faker and Passlib's CryptContext
fake = Faker()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load environment variables from .env file
load_dotenv()

# Pydantic Settings for Configuration (Ensure this matches your main application settings)
class TestSettings(Settings):
    class Config:
        env_prefix = "TEST_DB_"  # Prefix for test database environment variables

# Instantiate test settings
test_settings = TestSettings()

# Define the test database URL
TEST_DATABASE_URL = (
    f'postgresql://{test_settings.db_user}:'
    f'{test_settings.db_password}@{test_settings.db_host}:'
    f'{test_settings.db_port}/{test_settings.db_name}'
)


@pytest.fixture(scope='session')
def engine() -> Generator:
    """
    Creates a SQLAlchemy engine connected to the test database.
    """
    engine = create_engine(TEST_DATABASE_URL, echo=False)  # Set echo=True for verbose SQL logging
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope='session')
def SessionLocal(engine) -> Generator:
    """
    Creates a configured SQLAlchemy sessionmaker bound to the test engine.
    """
    Session = sessionmaker(bind=engine)
    yield Session
    # No teardown needed for sessionmaker


@pytest.fixture(scope='session', autouse=True)
def setup_database(engine):
    """
    Creates all tables before tests and drops them after all tests.
    """
    # Create tables
    Base.metadata.create_all(engine)
    yield
    # Drop tables
    # Base.metadata.drop_all(engine)


@pytest.fixture(scope='function')
def db_session(SessionLocal) -> Generator:
    """
    Creates a new database session for a test and rolls back after the test.
    """
    session = SessionLocal()
    try:
        yield session
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


@pytest.fixture(scope='function')
def test_user(db_session) -> User:
    """
    Creates and returns a test user in the database.
    """
    # Generate fake user data
    plain_password = fake.password(length=12)
    user_data = UserData(
        first_name=fake.first_name(),
        last_name=fake.last_name(),
        email=fake.unique.email(),
        username=fake.unique.user_name(),
        password=plain_password
    )
    # Hash the password with salt from test settings
    hashed_password = pwd_context.hash(user_data.password + test_settings.salt)
    
    # Create User instance
    user = User(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        email=user_data.email,
        username=user_data.username,
        password=hashed_password
    )
    
    # Add to session and commit
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)  # Refresh to get updated fields like id
    
    return user


@pytest.fixture(scope='session')
def fastapi_server():
    """
    Fixture to start the FastAPI server before E2E tests and stop it after tests complete.
    """
    # Start FastAPI app
    fastapi_process = subprocess.Popen(['python', 'main.py'])
    
    # Define the URL to check if the server is up
    server_url = 'http://127.0.0.1:8000/'
    
    # Wait for the server to start by polling the root endpoint
    timeout = 30  # seconds
    start_time = time.time()
    server_up = False
    
    print("Starting FastAPI server...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(server_url)
            if response.status_code == 200:
                server_up = True
                print("FastAPI server is up and running.")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    
    if not server_up:
        fastapi_process.terminate()
        raise RuntimeError("FastAPI server failed to start within timeout period.")
    
    yield
    
    # Terminate FastAPI server
    print("Shutting down FastAPI server...")
    fastapi_process.terminate()
    fastapi_process.wait()
    print("FastAPI server has been terminated.")


@pytest.fixture(scope="session")
def playwright_instance_fixture():
    """
    Fixture to manage Playwright's lifecycle.
    """
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright_instance_fixture):
    """
    Fixture to launch a browser instance.
    """
    browser = playwright_instance_fixture.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.fixture(scope="function")
def page(browser):
    """
    Fixture to create a new page for each test.
    """
    page = browser.new_page()
    yield page
    page.close()