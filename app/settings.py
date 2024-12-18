from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

# Pydantic Settings for Configuration
class Settings(BaseSettings):
    db_host: str
    db_user: str
    db_password: str
    db_name: str
    db_port: Optional[int] = 5432  # Default PostgreSQL port
    salt: str  # Additional salt for password hashing
    api_key: str  # Add this line

    class Config:
        env_file = ".env"
        env_prefix = "DB_"  # Prefix only for database-related environment variables
        fields = {
            'salt': {'env': 'SALT'},  # Map 'salt' to 'SALT' without the 'DB_' prefix
        }