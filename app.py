# --- app.py (Kisaan Mitra: ULTIMATE FINAL WORKING VERSION - Optimized) ---

# PRIMARY FASTAPI IMPORTS 
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from PIL import Image
import io
import os
import json
import asyncio 
from contextlib import asynccontextmanager 
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

# --- DUAL API IMPORTS ---
from openai import OpenAI
from google.genai.types import Content 
# ------------------------

# --- AUTH & DB IMPORTS ---
from databases import Database
from sqlalchemy import create_engine, Column, Integer, String, Boolean, MetaData, Table, DateTime
import datetime
import bcrypt
from jose import jwt, JWTError
import traceback 
# -------------------------

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") 

# ----------------------------------------------------
# 🐞 LOGGING: Initialization Check
# ----------------------------------------------------
print("\n[INIT] Starting Kisaan Mitra Backend setup...")
if API_KEY and len(API_KEY) > 10:
    print(f"[INIT] GEMINI_API_KEY loaded successfully.")
else:
    print("[CRITICAL] GEMINI_API_KEY is missing.")
if OPENAI_KEY and len(OPENAI_KEY) > 10:
    print("[INIT] OPENAI_API_KEY loaded successfully. (Dual API Mode)")
else:
    print("[INIT] WARNING: OpenAI Client will be skipped.")
    
# --- DB and Auth Setup ---
SECRET_KEY = os.environ.get("SECRET_KEY", "your-super-secret-key-replace-this") 
ALGORITHM = "HS256"
# FIX 1: AUTO-LOGOUT FIX - Set token to expire in 7 days (7 * 24 * 60 = 10080 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 10080 
DAILY_CREDIT_LIMIT = 20

DATABASE_URL = "sqlite:///./kisaan_mitra.db" 
database = Database(DATABASE_URL)
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(120), unique=True, index=True),
    Column("hashed_password", String(128)),
    Column("analysis_credits", Integer, default=DAILY_CREDIT_LIMIT), 
)

analysis_results = Table(
    "analysis_results", metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer), 
    Column("idea_summary", String(500)),
    Column("score", Integer),
    Column("created_at", DateTime, default=datetime.datetime.utcnow),
)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Auth Utilities (Optimized for threading)
def hash_password(password: str) -> str:
    # CPU-bound hashing
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=10)).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # CPU-bound verification
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    if 'sub' in to_encode:
        to_encode['sub'] = str(to_encode['sub'])
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Security(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub") 
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user_id_int = int(user_id) 
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired or invalid.")
    
    query = users.select().where(users.c.id == user_id_int)
    user = await database.fetch_one(query)

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    
    return {"id": user['id'], "email": user['email'], "analysis_credits": user['analysis_credits']}

# Models (Unchanged)
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    credits: int
class UserIn(BaseModel):
    email: EmailStr
    password: str
class Login(BaseModel):
    email: EmailStr
    password: str
class PredictionResponse(BaseModel):
    disease_name: str
    confidence: str
    remedy_suggestion: str
    initial_chat_message: str
class ChatContextRequest(BaseModel):
    session_id: str
    message: str
    context: str = "" 
class ChatContextResponse(BaseModel):
    reply: str
# ----------------------------------------------------

# ----------------------------------------------------
# 🐞 LOGGING: Lifespan (Startup)
# ----------------------------------------------------
GEMINI_CLIENT = None 
OPENAI_CLIENT = None 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEMINI_CLIENT, OPENAI_CLIENT
    print("[LIFESPAN] Starting up server components...")
    try:
        if API_KEY:
            GEMINI_CLIENT = genai.Client(api_key=API_KEY)
            
        if OPENAI_KEY:
            OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY)
            
        print(f"[LIFESPAN] Clients Ready: Gemini={bool(GEMINI_CLIENT)}, OpenAI={bool(OPENAI_CLIENT)}")
            
    except Exception as e:
        print(f"[LIFESPAN] CRITICAL ERROR: API Client initialization failed: {e}")
        
    metadata.create_all(engine)
    await database.connect()
    print("[LIFESPAN] Database connected and schema checked.")
    yield
    print("[LIFESPAN] Shutting down. Closing database connection.")
    await database.disconnect()
    
app = FastAPI(title="Kisaan-Mitra AI (Optimized)", lifespan=lifespan)

# CORS setup (Unchanged)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat Session Management (Unchanged)
chat_sessions = {}

# ----------------------------------------------------
# 🚀 SPEED OPTIMIZATION: System Instruction (Shortened)
# ----------------------------------------------------
def get_system_instruction(context: str) -> str:
    """System instruction ko chota kar diya hai taaki har chat call fast ho."""
    
    govt_schemes = """
    Latest Schemes: PM-KISAN (Rs. 6k), PMFBY (Crop Insurance), Soil Health Card, Krishi Udaan (Transport).
    """
    instruction = f"""
    Aap Kisaan Mitra, ek AI-Driven Farming Assistant hain. Aapka role sirf farming, crop health, aur agriculture-related sawalon ka jawab dena hai. Aapka jawab brief aur Hinglish mein hona chahiye.
    CRITICAL CONSTRAINT: Non-farming related sawalon ka jawab nahi dena hai.
    CONTEXT: User ka current crop/disease context: "{context}"
    LATEST GOVERNMENT SCHEMES (Must-Know): {govt_schemes}
    """
    return instruction
# ----------------------------------------------------

# ----------------------------------------------------
# --- AUTH & PREDICT ENDPOINTS (Optimized) ---
# ----------------------------------------------------

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserIn):
    print(f"[AUTH] SIGNUP attempt for email: {user.email}")
    query = users.select().where(users.c.email == user.email)
    existing_user = await database.fetch_one(query)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")

    # Optimized: Offload heavy CPU task (hashing) to a separate thread
    hashed_password = await asyncio.to_thread(hash_password, user.password)
    
    query = users.insert().values(email=user.email, hashed_password=hashed_password, analysis_credits=DAILY_CREDIT_LIMIT)
    last_record_id = await database.execute(query)
    
    new_user_query = users.select().where(users.c.id == last_record_id)
    new_user = await database.fetch_one(new_user_query)
    
    access_token = create_access_token(data={"sub": new_user['id']}) 
    print(f"[AUTH] SUCCESS: User {new_user['id']} signed up.")
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_id": new_user['id'], 
        "credits": new_user['analysis_credits']
    }

@app.post("/auth/login")
async def login_for_access_token(credentials: Login):
    print(f"[AUTH] LOGIN attempt for email: {credentials.email}")
    query = users.select().where(users.c.email == credentials.email)
    user = await database.fetch_one(query)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    
    # Optimized: Offload heavy CPU task (verification) to a separate thread
    is_valid = await asyncio.to_thread(verify_password, credentials.password, user.hashed_password)

    if not is_valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    
    access_token = create_access_token(data={"sub": user.id}) 
    print(f"[AUTH] SUCCESS: User {user.id} logged in.")
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user_id": user.id, 
        "credits": user.analysis_credits
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...), 
    session_id: str = Form("default_session"),
    current_user: dict = Depends(get_current_user) 
):
    user_id = current_user['id']
    print(f"\n[PREDICT] User {user_id} started prediction for session: {session_id}")
    
    # Prediction MUST use Gemini for Vision capability
    if not GEMINI_CLIENT:
         print("[PREDICT] ERROR: Gemini Client is NOT initialized.")
         raise HTTPException(status_code=500, detail="Gemini Client is not running for prediction.")

    try:
        # Step 1: Read and process image (Optimized for threading)
        print("[PREDICT] Step 1: Reading uploaded image data...")
        image_data = await file.read()
        # Optimized: Image processing (PIL) is CPU intensive, offload it
        image = await asyncio.to_thread(Image.open, io.BytesIO(image_data)) 
        print("[PREDICT] Step 1 Complete: Image loaded into PIL via threading.")
        
        prediction_prompt = """
        Is image mein jo crop leaf dikhaya gaya hai, uski disease ko identify karo. 
        Agar disease nahi hai, toh 'Healthy' mention karo.
        
        Output mein sirf ek JSON object dena hai:
        {{
            "disease_name": "Disease ka naam ya Healthy", 
            "confidence": "Prediction ki confidence level (e.g., High, Medium)",
            "remedy_suggestion": "Ek concise, practical remedy (Hinglish mein).",
            "initial_chat_message": "Ek welcoming message jo user ko next steps bataye."
        }}
        """

        # Step 2: Call Gemini Vision API (Blocking I/O, must be in a thread)
        print("[PREDICT] Step 2: Calling Gemini Vision API...")
        response = await asyncio.to_thread(
            GEMINI_CLIENT.models.generate_content,
            model='gemini-2.5-flash',
            contents=[prediction_prompt, image],
            config={"response_mime_type": "application/json"}
        )
        print(f"[PREDICT] Step 2 Complete: Gemini Response received.")
        
        # Step 3: Parse and validate response
        data = await asyncio.to_thread(json.loads, response.text)
        print(f"[PREDICT] Step 3: Parsed Disease: {data.get('disease_name', 'No Name')}")
        
        # Step 4: Save history and setup chat session (Asynchronous DB update)
        score = 90 if "Healthy" in data.get('disease_name', 'Unknown') else 50 
        await database.execute(analysis_results.insert().values(
             user_id=user_id, 
             idea_summary=f"Prediction: {data.get('disease_name', 'Unknown Disease')}",
             score=score,
        ))
        print(f"[PREDICT] Step 4 Complete: History saved.")
        
        full_initial_message = f"Prediction complete. {data.get('initial_chat_message')}"
        chat_sessions[session_id] = [{"role": "assistant", "content": full_initial_message}]
        
        print("[PREDICT] SUCCESS: Prediction flow finished.")
        return PredictionResponse(**data)
        
    except APIError as e:
        print(f"[PREDICT] CRITICAL ERROR: APIError occurred: {e}")
        raise HTTPException(status_code=500, detail="AI Service connect nahi ho paya. API Key check karein.")
    except Exception as e:
        print(f"[PREDICT] CRITICAL ERROR: Server Exception occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Server par koi technical problem hai.")


# Chat Endpoint (Dual API Fallback)
@app.post("/chat_context", response_model=ChatContextResponse)
async def chat_with_context(request: ChatContextRequest): 
    session_id = request.session_id
    user_message = request.message
    context = request.context
    
    print(f"\n[CHAT] Session {session_id} received message: '{user_message[:30]}...' (Context: {context})")
    
    if not GEMINI_CLIENT and not OPENAI_CLIENT:
         print("[CHAT] ERROR: No Chat service available.")
         raise HTTPException(status_code=500, detail="Koi bhi Chat service (Gemini/OpenAI) available nahi hai.")
         
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        print(f"[CHAT] Warning: New chat session created: {session_id}")
        
    history = chat_sessions[session_id]
    history.append({"role": "user", "content": user_message})
    
    system_instruction = get_system_instruction(context)

    reply_text = ""
    
    # 1. 🚀 SPEED UP ATTEMPT (OpenAI/Faster Model)
    if OPENAI_CLIENT:
        print("[CHAT] Attempt 1: Using OpenAI (Fastest Path)...")
        try:
            # Convert simple history to OpenAI message format
            openai_messages = [
                {"role": "system", "content": system_instruction}
            ] + [
                {"role": "user" if m['role'] == 'user' else "assistant", "content": m['content']}
                for m in history
            ]
            
            # API Call (Blocking I/O - must be in a thread)
            openai_response = await asyncio.to_thread(
                OPENAI_CLIENT.chat.completions.create,
                model="gpt-3.5-turbo", # FASTEST chat model choice
                messages=openai_messages
            )
            
            reply_text = openai_response.choices[0].message.content
            print("[CHAT] SUCCESS: Reply generated via OpenAI.")
            
        except Exception as e:
            print(f"[CHAT] WARNING: OpenAI failed ({e.__class__.__name__}). Falling back to Gemini.")
            traceback.print_exc() 
            pass # Fallback to Gemini

    # 2. 🛡️ FALLBACK TO GEMINI (Robustness)
    if not reply_text and GEMINI_CLIENT:
        print("[CHAT] Attempt 2: Using Gemini Client (Fallback)...")
        try:
            # Step 1: Convert dictionary history to Gemini Content Objects 
            gemini_contents = []
            for message in history:
                # FIX: Role Correction for Gemini API
                role_to_use = 'user' if message['role'] == 'user' else 'model' 
                
                gemini_contents.append(
                    Content(role=role_to_use, parts=[{'text': message['content']}])
                )
            
            # API Call (Blocking I/O - must be in a thread)
            gemini_response = await asyncio.to_thread(
                GEMINI_CLIENT.models.generate_content,
                model='gemini-2.5-flash',
                contents=gemini_contents, 
                config={"system_instruction": system_instruction}
            )
            
            reply_text = gemini_response.text
            print("[CHAT] SUCCESS: Reply generated via Gemini.")
            
        except APIError as e:
            error_detail = json.loads(e.message)['error']['message']
            print(f"[CHAT] CRITICAL ERROR: Gemini API failed: {error_detail}")
            raise HTTPException(status_code=500, detail=f"Chat service connect nahi ho paya. Error: {error_detail}")
        except Exception as e:
            print(f"[CHAT] CRITICAL ERROR: Chat processing failed (Gemini): {e}")
            raise HTTPException(status_code=500, detail="Chat processing mein galti hui.")
    
    if not reply_text:
        raise HTTPException(status_code=500, detail="Koi bhi AI service (OpenAI ya Gemini) reply nahi de paya.")


    # Step 3: Handle response and update session
    if not reply_text or reply_text.strip() == "":
        reply_text = "Mujhe aapka sawaal theek se samajh nahi aaya. Kya aap doobara pooch sakte hain?"
    
    # IMPORTANT: Store history for the next turn
    history.append({"role": "assistant", "content": reply_text}) 
    chat_sessions[request.session_id] = history[-10:] 
    
    return ChatContextResponse(reply=reply_text)

@app.get("/history")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    print(f"\n[HISTORY] Fetching history for user {user_id}")
    query = analysis_results.select().where(analysis_results.c.user_id == user_id).order_by(analysis_results.c.created_at.desc()).limit(10)
    history = await database.fetch_all(query) 
    print(f"[HISTORY] SUCCESS: Found {len(history)} items.")
    
    formatted_history = [
        {
            "date": item['created_at'].strftime("%Y-%m-%d %I:%M %p") if item['created_at'] else 'N/A', 
            "idea": item['idea_summary'],
            "score": item['score']
        }
        for item in history
    ]
    
    return {"history": formatted_history}

@app.delete("/history/clear", status_code=status.HTTP_200_OK)
async def clear_user_history(current_user: dict = Depends(get_current_user)):
    """Deletes all analysis history records for the current user."""
    user_id = current_user['id']
    print(f"\n[HISTORY] User {user_id} requested to clear history.")
    
    delete_query = analysis_results.delete().where(analysis_results.c.user_id == user_id)
    await database.execute(delete_query)
    
    print(f"[HISTORY] SUCCESS: History cleared for user {user_id}.")
    return {"message": "History cleared successfully."}