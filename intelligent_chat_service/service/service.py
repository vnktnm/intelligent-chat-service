from fastapi import FastAPI, Request
import time
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import os
from utils import logger
from utils.logger_utils import set_request_id
from controller import chat_router
import uuid


@asynccontextmanager
async def lifespan(app: FastAPI):
    # add any startup code here

    yield

    # add any shutdown code here
    print("Shutdown, Perform any cleanup if necessary...")


app = FastAPI(lifespan=lifespan, title="Intelligent Chat Service", version="1.0.0")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # create a request ID with UUID format
    request_id = request.headers.get("x-request-id", f"req-{uuid.uuid4()}")

    # Set the request ID in the context
    set_request_id(request_id)

    # log the request
    logger.info(
        f"Request Started: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
        },
    )

    response = await call_next(request)

    # calculate request duration
    duration = time.time() - start_time

    # log the response
    logger.info(
        f"Request Completed: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return response


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Intelligent Chat Service!"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(chat_router)
