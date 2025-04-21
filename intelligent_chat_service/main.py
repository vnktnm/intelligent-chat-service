from fastapi import FastAPI
from controller.graph_api import graph_router

app = FastAPI()

app.include_router(graph_router)
