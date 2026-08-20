from fastapi import FastAPI
from app.api.routes import router
from app.core.response import setup_exception_handlers

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",   # React/Next frontend
    "http://localhost:5173",   # Vite frontend
    "https://your-frontend-domain.com",  # deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


setup_exception_handlers(app)

app.include_router(router)