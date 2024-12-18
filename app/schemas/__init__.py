# Pydantic Model for User Data Validation
from pydantic import BaseModel, EmailStr


class UserData(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    username: str
    password: str  # Plain password for hashing