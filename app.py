# app.py - Sonify Podcast Generator (Stable Version)

import os
import uuid
import re
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pydub import AudioSegment
from google.cloud import texttospeech
import google.generativeai as genai
from google.oauth2 import service_account

# --- App & CORS Configuration ---
app = Flask(__name__)

# This list tells your backend that it's safe to accept requests
# from these specific web addresses.
origins = [
    "https://vermillion-otter-bfe24a.netlify.app",
    "https://statuesque-tiramisu-4b5936.netlify.app",
    "https://coruscating-hotteok-a5fb56.netlify.app",
    "https://www.mosaicdigital.ai",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    re.compile(r"https://.*\.netlify\.app"), # Allow all netlify subdomains
]
CORS(app, resources={r"/*": {"origins": origins}})

# --- Service Initialization Globals ---
db = None
bucket = None
tts_client = None
genai_model = None

CREDENTIALS_PATH = "/etc/secrets/firebase_service_account.json"

def initialize_services():
    """Initializes all external services using a secret file."""
    global db, bucket, tts_client, genai_model

    # Create two different credential objects from the same file
    try:
        print(f"Loading credentials from secret file: {CREDENTIALS_PATH}")
        # Credential for Firebase Admin SDK
        cred_firebase = credentials.Certificate(CREDENTIALS_PATH)
        # Credential for other Google Cloud services (like TTS)
        cred_gcp = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    except Exception as e:
        print(f"FATAL: Could not load credentials from {CREDENTIALS_PATH}. Error: {e}")
        raise e

    # Initialize Firebase
    if not firebase_admin._apps:
        try:
            print("Initializing Firebase...")
            project_id = os.environ.get('GCP_PROJECT_ID')
            firebase_admin.initialize_app(cred_firebase, { # Use the Firebase credential
                'projectId': project_id,
                'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
            })
            db = firestore.client()
            bucket = storage.bucket()
            print("Successfully connected to Firebase.")
        except Exception as e:
            print(f"FATAL: Could not connect to Firebase: {e}")
            raise e

    # Initialize Text-to-Speech
    if tts_client is None:
        try:
            print("Initializing Google Cloud Text-to-Speech client...")
            # Use the general Google Cloud credential
            tts_client = texttospeech.TextToSpeechClient(credentials=cred_gcp)
            print("Text-to-Speech client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize TTS client: {e}")
            raise e
            
    # Initialize Gemini
    if genai_model is None:
        try:
            print("Initializing Google Gemini model...")
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)
            genai_model = genai.GenerativeModel('gemini-1.5-pro-latest')
            print("Gemini model initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize Gemini model: {e}")
            raise e
        
# --- Celery Configuration ---
def make_celery(app):
    broker_url = os.environ.get('CELERY_BROKER_URL')
    if not broker_url:
        raise RuntimeError("CELERY_BROKER_URL environment variable is not set.")

    celery = Celery(app.import_name, backend=broker_url, broker=broker_url)
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                initialize_services()
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# --- Core Logic Functions ---
def generate_script_from_idea(topic, context, duration):
    print(f"Generating PODCAST script for topic: {topic}")
    prompt = (
        "You are a scriptwriter for a popular podcast. Your task is to write a script for two AI hosts, Trystan (male) and Saylor (female). "
        "The hosts are witty, charismatic, and engaging. The dialogue should feel natural, warm, and have a good back-and-forth conversational flow. "
        f"The topic is: '{topic}'. "
        f"Additional context: '{context}'. "
        f"The podcast should be approximately {duration} long. "
        "--- \n"
        "IMPORTANT INSTRUCTIONS: \n"
        "1.  Start each line with the speaker's tag, either '[Trystan]' or '[Saylor]'. \n"
        "2.  Alternate speakers for each line of dialogue. \n"
        "3.  Do NOT include any other text, directions, or formatting. \n"
        "4.  EXAMPLE: \n"
        "[Trystan] Welcome back to AI Insights! Today, we're tackling a huge topic: quantum computing. \n"
        "[Saylor] It sounds intimidating, but I promise we'll make it fun. Ready to dive in? \n"
        "[Trystan] Absolutely. So, at its core, what makes a quantum computer different from the one on your desk?"
    )
    response = genai_model.generate_content(prompt)
    print("Podcast script generated successfully.")
    return response.text

def parse_script(script_text):
    """Parses a script with named speaker tags into a list of (speaker, dialogue) tuples."""
    print("Parsing script...")
    pattern = re.compile(r'\[(Trystan|Saylor)\]\s*([^\n\[\]]*)')
    dialogue_parts = pattern.findall(script_text)
    print(f"Parsed {len(dialogue_parts)} dialogue parts.")
    return dialogue_parts

def generate_podcast_audio(script_text, output_filepath, voice_names):
    """Generates podcast audio by parsing a script and stitching the parts together."""
    print(f"Generating audio for voices: {voice_names}")
    dialogue_parts = parse_script(script_text)
    if not dialogue_parts:
        raise ValueError("The script is empty or could not be parsed. Cannot generate audio.")
    
    voice_map = {'Trystan': voice_names[0], 'Saylor': voice_names[1]}
    combined_audio = AudioSegment.empty()
    
    print("---- STARTING AUDIO SYNTHESIS LOOP ----")
    for i, (speaker_name, dialogue) in enumerate(dialogue_parts):
        dialogue = dialogue.strip()
        print(f"\n[PART {i+1}/{len(dialogue_parts)}] Speaker: {speaker_name}")
        print(f"[PART {i+1}] Dialogue to synthesize: '{dialogue}'")

        if not dialogue: 
            print(f"[PART {i+1}] SKIPPING: Dialogue is empty.")
            continue

        voice_name = voice_map.get(speaker_name)
        if not voice_name: 
            print(f"[PART {i+1}] SKIPPING: Could not find voice for speaker.")
            continue

        phonetic_dialogue = dialogue.replace("Saylor", "sailor")
        synthesis_input = texttospeech.SynthesisInput(text=phonetic_dialogue)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice_name.split('-')[0] + '-' + voice_name.split('-')[1],
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            sample_rate_hertz=24000
        )
        
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        
        print(f"[PART {i+1}] TTS API returned audio content size: {len(response.audio_content)} bytes")

        if not response.audio_content:
            print(f"[PART {i+1}] WARNING: TTS API returned empty audio. Skipping this part.")
            continue
            
        audio_chunk = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        print(f"[PART {i+1}] Pydub created audio chunk of duration: {len(audio_chunk)} ms")

        combined_audio += audio_chunk + AudioSegment.silent(duration=600)
    
    print("\n---- FINISHED AUDIO SYNTHESIS LOOP ----")
    print(f"Total combined audio duration before export: {len(combined_audio)} ms")

    if len(combined_audio) == 0:
        raise ValueError("Audio generation resulted in an empty file. All TTS requests may have failed.")

    combined_audio.export(output_filepath, format="mp3")
    print(f"Audio content successfully written to file '{output_filepath}'")
    return True

def _finalize_job(job_id, collection_name, local_audio_path, storage_path, generated_script=None, local_artwork_path=None):
    """Finalizes a job by uploading files and updating Firestore."""
    print(f"Finalizing job {job_id}...")
    
    # 1. Upload Audio File
    audio_blob = bucket.blob(storage_path)
    print(f"Uploading {local_audio_path} to {storage_path}...")
    audio_blob.upload_from_filename(local_audio_path)
    audio_blob.make_public()
    audio_url = audio_blob.public_url
    print(f"Audio upload complete. Public URL: {audio_url}")
    os.remove(local_audio_path)
    
    # Prepare the data for the database update
    update_data = {
        'status': 'complete', 
        'url': audio_url, 
        'completed_at': firestore.SERVER_TIMESTAMP
    }
    if generated_script:
        update_data['generated_script'] = generated_script

    # 2. Upload Artwork File (if it was created)
    if local_artwork_path:
        artwork_storage_path = f"podcasts/artwork/{os.path.basename(local_artwork_path)}"
        artwork_blob = bucket.blob(artwork_storage_path)
        print(f"Uploading {local_artwork_path} to {artwork_storage_path}...")
        artwork_blob.upload_from_filename(local_artwork_path)
        artwork_blob.make_public()
        artwork_url = artwork_blob.public_url
        print(f"Artwork upload complete. Public URL: {artwork_url}")
        os.remove(local_artwork_path)
        # Add the artwork URL to our database update
        update_data['artwork_url'] = artwork_url

    # 3. Update Firestore with all the new data
    db.collection(collection_name).document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "url": audio_url}

def generate_artwork_for_topic(topic, job_id):
    """Generates podcast cover art using an image model."""
    print(f"Generating artwork for topic: {topic}")
    try:
        # This assumes you have the 'Vertex AI User' role on your service account
        from google.cloud import aiplatform
        # This line has been corrected
        from google.cloud.aiplatform.gapic.preview import image_generation_service_client as igs_client

        # Configure the image generation client
        api_endpoint = "us-central1-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        client = igs_client.ImageGenerationServiceClient(client_options=client_options)

        prompt = (
            f"Podcast cover art for a podcast about '{topic}'. "
            f"Digital art, vibrant, modern, aesthetically pleasing, no text."
        )

        # Generate the image
        response = client.generate_images(
            parent=f"projects/{os.environ.get('GCP_PROJECT_ID')}/locations/us-central1",
            prompt=prompt,
            number_of_images=1
        )

        if not response.images:
            raise Exception("Image generation returned no images.")

        # Save the image to a temporary file
        artwork_filepath = f"{job_id}_artwork.png"
        image_bytes = response.images[0].image_bytes
        with open(artwork_filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"Artwork successfully written to file '{artwork_filepath}'")
        return artwork_filepath
    except Exception as e:
        print(f"WARNING: Artwork generation failed: {e}")
        return None

# --- Celery Task Definitions ---
@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration, voices):
    print(f"WORKER: Started PODCAST job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    audio_filepath = f"{job_id}.mp3"
    artwork_filepath = None # Initialize artwork path as None

    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP, 'voices': voices})
        
        # --- CORRECTED LOGIC ---
        # 1. Generate the script
        original_script = generate_script_from_idea(topic, context, duration)
        
        # 2. Generate the artwork (this is the new step)
        artwork_filepath = generate_artwork_for_topic(topic, job_id) 
        
        # 3. Generate the audio
        if not generate_podcast_audio(original_script, audio_filepath, voices): 
            raise Exception("Audio generation failed.")
            
        # 4. Finalize the job with both audio and artwork files
        return _finalize_job(
            job_id, 
            'podcasts', 
            audio_filepath, 
            f"podcasts/{audio_filepath}", 
            generated_script=original_script, 
            local_artwork_path=artwork_filepath
        )
    except Exception as e:
        print(f"ERROR in podcast task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        # Clean up both temporary files on failure
        if os.path.exists(audio_filepath): os.remove(audio_filepath)
        if artwork_filepath and os.path.exists(artwork_filepath): os.remove(artwork_filepath)
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
@app.before_request
def before_first_request_func():
    initialize_services()

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Sonify API! The server is running."})

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context']):
        return jsonify({"error": "topic and context are required"}), 400
    job_id = str(uuid.uuid4())
    voices = data.get('voices', ['en-US-Chirp3-HD-Iapetus', 'en-US-Chirp3-HD-Leda'])
    generate_podcast_from_idea_task.delay(job_id, data['topic'], data['context'], data.get('duration', '5 minutes'), voices)
    return jsonify({"message": "Podcast generation has been queued!", "job_id": job_id}), 202

@app.route("/podcast-status/<job_id>", methods=["GET"])
def get_podcast_status(job_id):
    try:
        doc_ref = db.collection('podcasts').document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

