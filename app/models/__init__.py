from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import uuid

Base = declarative_base()

# Define the SQLAlchemy User model with UUID as primary key
class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # UUID primary key
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)  # Hashed password
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship with Calculation
    calculations = relationship("Calculation", back_populates="user")

    def __repr__(self):
        return f"<User(name={self.first_name} {self.last_name}, email={self.email})>"

# Define the SQLAlchemy Calculation model
class Calculation(Base):
    __tablename__ = 'calculations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # UUID primary key
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)  # Foreign key to User
    type = Column(String(50), nullable=False)  # Type of calculation (e.g., "addition", "subtraction")
    value_a = Column(Float, nullable=False)  # First value
    value_b = Column(Float, nullable=False)  # Second value
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship with User
    user = relationship("User", back_populates="calculations")

    def __repr__(self):
        return f"<Calculation(type={self.type}, value_a={self.value_a}, value_b={self.value_b})>"