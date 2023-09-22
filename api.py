from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from exa import Kosmos


app = FastAPI()
kosmos = Kosmos()

class PhraseImage(BaseModel):
    phrase: str
    image_url: str

class Image(BaseModel):
    image_url: str

@app.post("/kosmos/multimodal_grounding/")
async def multimodal_grounding(data: PhraseImage):
    return kosmos.multimodal_grounding(data.phrase, data.image_url)

@app.post("/kosmos/referring_expression_comprehension/")
async def referring_expression_comprehension(data: PhraseImage):
    return kosmos.referring_expression_comprehension(data.phrase, data.image_url)

@app.post("/kosmos/referring_expression_generation/")
async def referring_expression_generation(data: PhraseImage):
    return kosmos.referring_expression_generation(data.phrase, data.image_url)

@app.post("/kosmos/grounded_vqa/")
async def grounded_vqa(data: PhraseImage):
    return kosmos.grounded_vqa(data.phrase, data.image_url)

@app.post("/kosmos/grounded_image_captioning/")
async def grounded_image_captioning(data: Image):
    return kosmos.grounded_image_captioning(data.image_url)

@app.post("/kosmos/grounded_image_captioning_detailed/")
async def grounded_image_captioning_detailed(data: Image):
    return kosmos.grounded_image_captioning_detailed(data.image_url)

@app.post("/kosmos/generate_boxes/")
async def generate_boxes(data: PhraseImage):
    return kosmos.generate_boxees(data.phrase, data.image_url)

@app.get("/kosmos/inference/")
async def inference():
    return {"message": "Kosmos is ready for inference"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to Kosmos API. Please use the appropriate endpoint to perform tasks."}