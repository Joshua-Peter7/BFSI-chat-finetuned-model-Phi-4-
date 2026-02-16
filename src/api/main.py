"""FastAPI Backend for BFSI Conversational AI"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.orchestrator import Orchestrator
from src.core.formatter.response_formatter import ResponseFormatter

# Project root (bfsI SYSTEM)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_kb_dataset() -> list:
    """Load all KB datasets from data/raw (alpaca format: input, output, instruction)."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    combined = []
    if not raw_dir.exists():
        return combined
    for path in sorted(raw_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("input"):
                        combined.append(item)
            elif isinstance(data, dict) and data.get("input"):
                combined.append(data)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return combined


app = FastAPI(
    title="BFSI Conversational AI",
    description="Enterprise-grade conversational AI for BFSI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()
formatter = ResponseFormatter()

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def startup_load_kb():
    """Load your KB data into the vector store so Tier 1 and similarity search use it."""
    dataset = _load_kb_dataset()
    if dataset:
        try:
            orchestrator.intent_engine.index_dataset(dataset)
            print(f"Loaded {len(dataset)} KB examples into intent/similarity engine.")
        except Exception as e:
            print(f"Warning: KB indexing failed: {e}")
    else:
        print("No KB dataset found in data/raw/*.json (need 'input' / 'output' / 'instruction'). Add data and restart.")


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"


class QueryResponse(BaseModel):
    response_id: str
    query: str
    response: str
    intent: str
    confidence: float
    tier_used: int
    safe: bool
    processing_time_ms: float


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"

    if not html_path.exists():
        return """
        <html>
            <head><title>BFSI AI</title></head>
            <body>
                <h1>BFSI Conversational AI</h1>
                <p>API is running. Frontend not found.</p>
                <p>Go to <a href="/docs">/docs</a> for API documentation</p>
            </body>
        </html>
        """

    return html_path.read_text()


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        result = orchestrator.process(
            query=request.query,
            session_id=request.session_id
        )

        formatted = formatter.format(result)

        return QueryResponse(
            response_id=formatted.response_id,
            query=formatted.query,
            response=formatted.response,
            intent=formatted.intent,
            confidence=formatted.confidence,
            tier_used=formatted.tier_used,
            safe=formatted.safe,
            processing_time_ms=formatted.processing_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "system": "BFSI Conversational AI",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
