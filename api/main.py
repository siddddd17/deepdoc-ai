from fastapi import FastAPI, UploadFile, File, Form, HttpException, Request
from fastapi.responses import HTMLResponse, JsonResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DeepDoc AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_class=JsonResponse)
async def health_check():
    return JsonResponse(content={"status": "ok"}, status_code=200)

@app.post("/analyse", response_class=JsonResponse)
async def analyse_documents(file: UploadFile = File(...)) -> Any: 
    try: 
        pass 
    except HttpException as http_exc:
        raise http_exc
    except Exception as e:
        raise HttpException(status_code=500, detail=f"Analysis failed") from e

@app.pos("/compare", response_class=JsonResponse)
async def compare_documents(reference_file: UploadFile = File(...), actual_file: UploadFile = File(...)) -> Any:
    try:
        pass 
    except HttpException as http_exc:
        raise http_exc
    except Exception as e:
        raise HttpException(status_code=500, detail=f"Comparison failed") from e

@app.post('/chat/index')
async def chat_with_documents(uploaded_files: list[UploadFile] = File(...), user_query: str = Form(...)) -> Any:
    try:
        pass 
    except HttpException as http_exc:
        raise http_exc
    except Exception as e:
        raise HttpException(status_code=500, detail=f"Chat with documents failed") from e

@app.post('/chat/query')
async def chat_with_query(session_id: str = Form(...), user_query: str = Form(...)) -> Any:
    try:
        pass 
    except HttpException as http_exc:
        raise http_exc
    except Exception as e:  
        raise HttpException(status_code=500, detail=f"Chat with query failed") from e
        
