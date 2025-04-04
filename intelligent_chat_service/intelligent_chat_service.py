import uvicorn
from dotenv import load_dotenv
from config import SERVICE_HOST, SERVICE_PORT

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "service:app",
        host=SERVICE_HOST,
        port=int(SERVICE_PORT),
        reload=True,
    )
