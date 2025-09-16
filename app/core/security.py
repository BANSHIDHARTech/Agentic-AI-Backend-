from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# Configuration - Move these to environment variables in production
SECRET_KEY = "your-secret-key-here"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# This will be used as a dependency in protected routes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        if username is None or user_id is None:
            raise credentials_exception
        token_data = TokenData(username=username, user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    # In a real application, you would validate the user against your database here
    # For now, we'll just return the token data
    return token_data

# Simple user model for demonstration
class User(BaseModel):
    username: str
    user_id: str
    disabled: Optional[bool] = None

# This is a mock user database - replace with your actual user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "user_id": "user123",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}

async def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or password != "secret":  # In production, use proper password hashing
        return False
    return User(**user)

def get_password_hash(password: str):
    # In production, use a proper password hashing function like passlib
    return password  # This is just for demonstration

def verify_password(plain_password: str, hashed_password: str):
    # In production, use passlib to verify hashed passwords
    return plain_password == hashed_password
