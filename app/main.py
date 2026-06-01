from fastapi import FastAPI

app = FastAPI(
    title="2026 EXPO AI Server",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "2026 EXPO AI Server Running"
    }