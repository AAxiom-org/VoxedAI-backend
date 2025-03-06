# Refactoring

## Phase 1: Fix the ingestion endpoint

All this can be done in one class that takes in the filetype and file path:

- [ ] **Audio Processing Module**

  - Handles: audio/mpeg, audio/wav, audio/mp4, audio/webm
  - Functionality: Audio transcription to convert spoken content to text

- [ ] **Document Extraction Module**

  - Handles: application/pdf, text/plain, text/markdown, text/csv, application/vnd.openxmlformats-officedocument.wordprocessingml.document
  - Functionality: Extract text content while preserving structure
  - Need to use image to text for pdf files aswell (maybe have some way to determine if necessary)

- [ ] **Spreadsheet Processing Module**

  - Handles: text/csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
  - Functionality: Extract tabular data and convert to textual representation

- [ ] **Presentation Processing Module**

  - Handles: application/vnd.openxmlformats-officedocument.presentationml.presentation
  - Functionality: Extract text from slides and capture text from embedded images using image-to-text

- [ ] **Image Analysis Module**

  - Handles: image/jpeg, image/png, image/gif, image/webp
  - Functionality: Image-to-text conversion to generate descriptions of visual content

- [ ] **Video Processing Module**

  - Handles: video/mp4, video/webm
  - Functionality: Extract audio track for transcription and potentially capture frame-by-frame images for image-to-text

- [ ] **Text Normalization and Indexing Module**
  - Takes output from all other modules and prepares it for embedding/vectorization
  - Handles consistent formatting, cleaning, and chunking across all sources

### Phase 1 notes:

```
For images and embedded images (in pdf or powerpoint, or etc.) use google vision. Use PyMuPDF for image and text extraction from pdf. Use python-docx to extract text from Word (.docx) documents. use markdown to convert Markdown to plain text. Use pandas for csv. Use openpyxl for excel .xlsx and xlrd for .xls. Use python-pptx for powerpoints .pptx and use google vision for nested images. Use ffmpeg-python to convert video to audio and OpenCV to extract frames from video for google vision.

Use whisper+ffmpeg for video/audio formats:
import whisper
import subprocess

def convert_audio(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ac', '1',
        '-ar', '16000',
        output_file
    ]

    subprocess.run(command, check=True)

convert_audio('input.*', 'output.wav')
model = whisper.load_model("tiny")
result = model.transcribe('output.wav')
print(result["text"])

Use gemini flash for image to text using visual reasoning:
local files:
from google import genai
import os
import PIL.Image


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


image_path = "chess.png"
image = PIL.Image.open(image_path)


response = client.models.generate_content(
   model="gemini-2.0-flash",
   contents=["What is this image?", image])


print(response.text)

Base64 encoded images
from google import genai
from google.genai import types

import requests

image_path = "https://goo.gle/instrument-img"
image = requests.get(image_path)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What is this image?",
              types.Part.from_bytes(data=image.content, mime_type="image/jpeg")])

print(response.text)

Multiple images
from google import genai
from google.genai import types

import pathlib
import PIL.Image

image_path_1 = "path/to/your/image1.jpeg"  # Replace with the actual path to your first image
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to your second image

image_url_1 = "https://goo.gle/instrument-img" # Replace with the actual URL to your third image

pil_image = PIL.Image.open(image_path_1)

b64_image = types.Part.from_bytes(
    data=pathlib.Path(image_path_2).read_bytes(),
    mime_type="image/jpeg"
)

downloaded_image = requests.get(image_url_1)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What do these images have in common?",
              pil_image, b64_image, downloaded_image])

print(response.text)

Large image payloads
When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files.
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

img_path = "/path/to/Cajun_instruments.jpg"
file_ref = client.files.upload(file=img_path)
print(f'{file_ref=}')

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What can you tell me about these instruments?",
              file_ref])

print(response.text)

accepted inputs:
'audio/mpeg'::text,
'audio/wav'::text,
'audio/mp4'::text,
'audio/webm'::text,
'application/pdf'::text,
'text/plain'::text,
'text/markdown'::text,
'text/csv'::text,
'application/vnd.openxmlformats-officedocument.presentationml.presentation'::text,
'application/vnd.openxmlformats-officedocument.wordprocessingml.document'::text,
'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'::text,
'image/jpeg'::text,
'image/png'::text,
'image/gif'::text,
'image/webp'::text,
'video/mp4'::text,
'video/webm'::text
```

## Phase 2: Fix the query endpoint

- [ ] Ensure that it takes in the list of files to include in the RAG (given by path)
- [ ] Only include files listed in the params for RAG
- [ ] Send the RAG results back when ready
- [ ] After streaming send recap of all contents back
- [ ] After all data is sent upsert the data into the supabase rag_output table

## Phase 3: Documentation

- [ ] Ensure the readme explains exactly how each endpoint works including inputs and outputs
