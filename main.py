import os
import joblib
import logging
import threading
import pandas as pd
import pdfplumber
import httpx
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# Some PDFs contain invalid drawing-color instructions. pdfplumber only needs
# their text here, so avoid logging those non-fatal rendering warnings.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from secret_api_keys import groq_api_key
os.environ['GROQ_API_KEY'] = groq_api_key

PDF_PATH = "Medical_Book.pdf"

VECTORSTORE_PATH = "vectorstore"

model_artifact = joblib.load("health_risk_model.joblib")
model = model_artifact["model"]
label_encoder = model_artifact["label_encoder"]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=25
)

index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
pkl_file = os.path.join(VECTORSTORE_PATH, "index.pkl")
vectorstore = None
vectorstore_lock = threading.Lock()


def get_vectorstore():
    """Load the saved index, or create it once when the RAG endpoint is used."""
    global vectorstore

    if vectorstore is not None:
        return vectorstore

    with vectorstore_lock:
        if vectorstore is not None:
            return vectorstore

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            logger.info("Loading saved FAISS vector store.")
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return vectorstore

        logger.info("No vector store found. Creating one from %s; this happens only once.", PDF_PATH)
        docs = []
        with pdfplumber.open(PDF_PATH) as pdf:
            total_pages = len(pdf.pages)
            for number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    docs.extend(splitter.split_text(page_text))
                if number % 50 == 0 or number == total_pages:
                    logger.info("Processed PDF page %d/%d (%d chunks).", number, total_pages, len(docs))

        if not docs:
            raise RuntimeError(f"No extractable text was found in {PDF_PATH}.")

        logger.info("Creating embeddings for %d text chunks.", len(docs))
        vectorstore = FAISS.from_texts(docs, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info("Vector store saved to %s.", VECTORSTORE_PATH)
        return vectorstore

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)


class HealthRequest(BaseModel):

    Age: int
    Gender: str
    BMI: float
    SmokingStatus: str
    AlcoholConsumptionPerWeek: float
    PhysicalActivityHoursPerWeek: float
    SleepDurationHoursPerDay: float
    ChronicDiseaseHistory: str
    StressLevel: int

    question: str


@app.get("/")
def home():

    return {
        "message": "Health Assistant Running"
    }

@app.post("/health-assessment")

def health_assessment(data: HealthRequest):


    df = pd.DataFrame([{

        "Age": data.Age,
        "Gender": data.Gender,
        "BMI": data.BMI,
        "Smoking Status": data.SmokingStatus,
        "Alcohol Consumption (per week)": data.AlcoholConsumptionPerWeek,
        "Physical Activity (hours/week)": data.PhysicalActivityHoursPerWeek,
        "Sleep Duration (hours/day)": data.SleepDurationHoursPerDay,
        "Chronic Disease History": data.ChronicDiseaseHistory,
        "Stress Level (1-10)": data.StressLevel
    }])

    prediction = model.predict(df)[0]
    risk_level = label_encoder.inverse_transform([prediction])[0]



    vector_store = get_vectorstore()


    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

        template="""
You are a health assessment assistant.
Answer using the context.Using the context and the applicant information, answer the user's question.
If question is not related to the context say:
"I don't know the answer to that question based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""
    )


    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),

        chain_type="stuff",

        chain_type_kwargs={
            "prompt": prompt_template
        },

    )

    final_query = f"""

Applicant Information:
{df.to_dict(orient='records')[0]}

Risk Level:
{risk_level}

User Question:
{data.question}
"""

    result = qa_chain.invoke({
        "query": final_query
    })


    return {

        "risk": risk_level,

        "answer": result["result"],

    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



from secret_api_keys import geocoding_api_key
os.environ['GEOCODING_API_KEY'] = geocoding_api_key


from secret_api_keys import geoapify_api_key
os.environ['GEOAPIFY_API_KEY'] = geoapify_api_key

class HospitalRequest(BaseModel):
    location: str
    radius: float = 5000
    max_results: int = 5

class HospitalRequest(BaseModel):
    location: str
    radius: int = 5000
    max_results: int = 5


# ---------------------------------------------------------
# Geocode Location
# ---------------------------------------------------------

async def geocode_location(location: str):

    if not geocoding_api_key:
        raise HTTPException(
            status_code=500,
            detail="Geocoding API key is not configured"
        )

    url = "https://geocode.maps.co/search"

    params = {
        "q": location,
        "api_key": geocoding_api_key,
        "format": "json"
    }

    async with httpx.AsyncClient(timeout=20) as client:

        response = await client.get(
            url,
            params=params
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Geocoding API error: {response.text}"
        )

    results = response.json()

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location}' could not be found"
        )

    latitude = float(results[0]["lat"])
    longitude = float(results[0]["lon"])

    return latitude, longitude


# ---------------------------------------------------------
# Find Nearby Hospitals
# ---------------------------------------------------------

@app.post("/nearest-hospitals")
async def nearest_hospitals(data: HospitalRequest):

    if not geoapify_api_key:
        raise HTTPException(
            status_code=500,
            detail="Geoapify API key is not configured"
        )

    # Step 1: Convert location name → coordinates
    latitude, longitude = await geocode_location(
        data.location
    )

    # Step 2: Search nearby hospitals
    url = "https://api.geoapify.com/v2/places"

    params = {
        "categories": "healthcare",
        "filter": f"circle:{longitude},{latitude},{data.radius}",
        "bias": f"proximity:{longitude},{latitude}",
        "limit": min(data.max_results, 20),
        "lang": "en",
        "apiKey": geoapify_api_key
    }

    async with httpx.AsyncClient(timeout=20) as client:

        response = await client.get(
            url,
            params=params
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Geoapify API error: {response.text}"
        )

    result = response.json()

    hospitals = []

    for feature in result.get("features", []):

        properties = feature.get("properties", {})

        # Keep only healthcare places that have a name
        name = properties.get("name")

        if not name:
            continue

        hospitals.append({
            "name": name,
            "address": properties.get("formatted"),
            "city": properties.get("city"),
            "state": properties.get("state"),
            "postcode": properties.get("postcode"),
            "distance_meters": properties.get("distance"),
            "latitude": properties.get("lat"),
            "longitude": properties.get("lon"),
            "place_id": properties.get("place_id")
        })

    return {
        "location": data.location,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude
        },
        "search_radius_meters": data.radius,
        "hospitals_found": len(hospitals),
        "hospitals": hospitals
    }