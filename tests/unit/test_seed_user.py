import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from app.models import Base, User
from seed_user import generate_fake_user, hash_password, seed_users, Settings

# Define a test database URL (use an in-memory SQLite database for testing)
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create a SQLAlchemy engine and session for testing
engine = create_engine(TEST_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Initialize the database
Base.metadata.create_all(bind=engine)

@pytest.fixture(scope='function')
def db_session():
    """
    Creates a new database session for a test and rolls back after the test.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()



def test_hash_password():
    plain_password = "password123"
    salt = "somesalt"
    hashed_password = hash_password(plain_password, salt)
    
    assert hashed_password != plain_password + salt
    assert len(hashed_password) > len(plain_password + salt)