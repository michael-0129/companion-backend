from fastapi import APIRouter, HTTPException
from app.schemas import AuthLoginRequest, AuthLoginResponse
from fastapi import status

router = APIRouter()

@router.post("/login", response_model=AuthLoginResponse)
def login(request: AuthLoginRequest):
    # Hardcoded credentials
    if request.username == "michael" and request.password == "Michael123!@":
        # In production, generate a real JWT. Here, return a dummy token.
        return AuthLoginResponse(
            access_token="dummy-token-for-michael",
            token_type="bearer",
            message="Login successful."
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password."
    ) 