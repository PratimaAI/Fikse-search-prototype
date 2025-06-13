from fastapi import FastAPI, Query
from search import search

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to FikseSearch API"}

@app.get("/search")
def search_api(q: str = Query(..., description="Search query"), k: int = 3):
    results = search(q, k)
    return {"query": q, "results": results}
