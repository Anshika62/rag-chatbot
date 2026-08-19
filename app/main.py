from fastapi import FastAPI
from app.api.routes import router
from app.core.response import setup_exception_handlers


app = FastAPI()

setup_exception_handlers(app)

app.include_router(router)