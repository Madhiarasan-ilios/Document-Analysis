import os
import boto3
import streamlit as st
import vertexai
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.cloud.vision_v1.types import Image as VisionImage
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv
import fitz  # PyMuPDF

# --- Load Environment Variables ---
load_dotenv()

# --- Environment Setup ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "iliosdigital-ai-poc-0127591fe3ba.json"
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Vertex AI & Gemini Setup ---
PROJECT_ID = "iliosdigital-ai-poc"
LOCATION = "global"
vertexai.init(project=PROJECT_ID, location=LOCATION)
gemini_model = GenerativeModel("gemini-2.5-pro")

# --- Google Cloud Vision Setup ---
vision_client = ImageAnnotatorClient(client_options={"api_key": API_KEY})

AWS_REGION = "ap-south-1"

translate_client = boto3.client(
    "translate",
    region_name=AWS_REGION
)

chat_model = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.0,
    region_name=AWS_REGION
)

translation_prompt_template = PromptTemplate(
    input_variables=["text_to_translate"],
    template="""
You are an expert translator. The following text needs to be translated from Tamil to English.

Text to Translate: "{text_to_translate}"

Provide a refined, natural, and contextually accurate English translation.
Respond only with the refined English text.
"""
)
translation_chain = RunnableSequence(translation_prompt_template | chat_model)

# --- Core Functions ---

def process_file(file):
    """Process uploaded file (PDF or image) and return list of (page_name, image_bytes)."""
    processed_images = []
    if file.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            image_bytes = pix.tobytes(output="png")
            processed_images.append((f"Page {page_num + 1}", image_bytes))
        doc.close()
    else:
        processed_images.append(("Image", file.read()))
    return processed_images

def extract_and_translate(image_name, image_bytes):
    """Perform OCR → Gemini refinement → Translation pipeline."""
    # Step 1: Initial OCR
    vision_image = VisionImage(content=image_bytes)
    response = vision_client.text_detection(image=vision_image)
    initial_text = response.text_annotations[0].description if response.text_annotations else ""

    # Step 2: Gemini refinement
    gemini_image = Part.from_data(image_bytes, mime_type="image/png")
    prompt_parts = [
        "You are a highly accurate OCR model. Extract all readable text from this image. "
        "The following is a preliminary text extraction: " + initial_text + ". "
        "Use this as a reference but correct any errors, merge broken words, and provide the most accurate, complete text. "
        "Return only the final text, no explanations.",
        gemini_image
    ]
    gemini_response = gemini_model.generate_content(prompt_parts)
    refined_text = gemini_response.text

    # Step 3: Translation
    try:
        aws_translate_response = translate_client.translate_text(
            Text=refined_text,
            SourceLanguageCode="auto",
            TargetLanguageCode="en"
        )
        aws_translated_text = aws_translate_response["TranslatedText"]

        refined_translation_result = translation_chain.invoke({
            "text_to_translate": refined_text
        })
        final_translation = refined_translation_result.content
    except Exception as e:
        final_translation = f"Translation failed: {e}"

    return refined_text, final_translation

# --- Streamlit UI ---

st.set_page_config(page_title="Tamil → English OCR & Translator", layout="wide")
st.title("📖 Tamil → English OCR & Translator")

uploaded_files = st.file_uploader(
    "Upload PDF or Image(s)", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_translations = []
    for uploaded_file in uploaded_files:
        st.subheader(f"📂 File: {uploaded_file.name}")
        images_to_process = process_file(uploaded_file)

        for image_name, image_bytes in images_to_process:
            with st.spinner(f"Processing {image_name}..."):
                extracted_text, translated_text = extract_and_translate(image_name, image_bytes)

            st.markdown(f"### {image_name}")
            with st.expander("📌 Extracted Tamil Text"):
                st.write(extracted_text)

            with st.expander("🌍 English Translation"):
                st.write(translated_text)

            all_translations.append(f"--- {uploaded_file.name} - {image_name} ---\n{translated_text}\n")

    # Download button
    if all_translations:
        full_text = "\n\n".join(all_translations)
        st.download_button(
            label="⬇️ Download Full Translated Text",
            data=full_text,
            file_name="translated_texts.txt",
            mime="text/plain"
        )
