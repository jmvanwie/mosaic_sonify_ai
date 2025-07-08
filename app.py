# app.py - Sonify Podcast Generator (Final Version)

import os
import uuid
import re
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pydub import AudioSegment
from google.cloud import texttospeech
import google.generativeai as genai
from google.oauth2 import service_account
import openai

# --- App & CORS Configuration ---
app = Flask(__name__)

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

initialize_services()

# --- Service Initialization Globals ---
db = None
bucket = None
tts_client = None
genai_model = None

CREDENTIALS_PATH = "/etc/secrets/firebase_service_account.json"

def initialize_services():
    """Initializes all external services using a secret file."""
    global db, bucket, tts_client, genai_model

    try:
        print(f"Loading credentials from secret file: {CREDENTIALS_PATH}")
        cred_firebase = credentials.Certificate(CREDENTIALS_PATH)
        cred_gcp = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    except Exception as e:
        print(f"FATAL: Could not load credentials from {CREDENTIALS_PATH}. Error: {e}")
        raise e

    if not firebase_admin._apps:
        try:
            print("Initializing Firebase...")
            project_id = os.environ.get('GCP_PROJECT_ID')
            firebase_admin.initialize_app(cred_firebase, {
                'projectId': project_id,
                'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
            })
            db = firestore.client()
            bucket = storage.bucket()
            print("Successfully connected to Firebase.")
        except Exception as e:
            print(f"FATAL: Could not connect to Firebase: {e}")
            raise e

    if tts_client is None:
        try:
            print("Initializing Google Cloud Text-to-Speech client...")
            tts_client = texttospeech.TextToSpeechClient(credentials=cred_gcp)
            print("Text-to-Speech client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize TTS client: {e}")
            raise e
            
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
    
    for i, (speaker_name, dialogue) in enumerate(dialogue_parts):
        dialogue = dialogue.strip()
        if not dialogue: continue

        voice_name = voice_map.get(speaker_name)
        if not voice_name: continue

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
        
        if not response.audio_content:
            continue
            
        audio_chunk = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        combined_audio += audio_chunk + AudioSegment.silent(duration=600)

    if len(combined_audio) == 0:
        raise ValueError("Audio generation resulted in an empty file. All TTS requests may have failed.")

    combined_audio.export(output_filepath, format="mp3")
    print(f"Audio content successfully written to file '{output_filepath}'")
    return True

# --- NEW, SIMPLER ARTWORK AND FINALIZE FUNCTIONS ---

def generate_artwork_for_topic(topic):
    """Generates podcast cover art using OpenAI's DALL-E 3 model."""
    print(f"Generating artwork for topic '{topic}' using DALL-E 3...")
    try:
        # The OpenAI client automatically uses the OPENAI_API_KEY environment variable
        client = openai.OpenAI()
        
        prompt = (
            f"Digital art of a podcast cover for a show about '{topic}'. "
            f"Vibrant, modern, aesthetically pleasing, no text."
        )

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        artwork_url = response.data[0].url
        if not artwork_url:
            raise Exception("DALL-E 3 did not return an image URL.")
            
        print(f"Artwork generated successfully. URL: {artwork_url}")
        return artwork_url
    except Exception as e:
        print(f"WARNING: DALL-E 3 artwork generation failed: {e}")
        return None

def _finalize_job(job_id, collection_name, local_audio_path, storage_path, generated_script=None, artwork_url=None):
    """Finalizes a job by uploading the audio file and updating Firestore."""
    print(f"Finalizing job {job_id}...")
    
    # 1. Upload Audio File
    audio_blob = bucket.blob(storage_path)
    print(f"Uploading {local_audio_path} to {storage_path}...")
    audio_blob.upload_from_filename(local_audio_path)
    audio_blob.make_public()
    audio_url = audio_blob.public_url
    print(f"Audio upload complete. Public URL: {audio_url}")
    os.remove(local_audio_path)
    
    # 2. Prepare the data for the database update
    update_data = {
        'status': 'complete', 
        'url': audio_url, 
        'completed_at': firestore.SERVER_TIMESTAMP
    }
    if generated_script:
        update_data['generated_script'] = generated_script
    if artwork_url:
        update_data['artwork_url'] = artwork_url # Add the artwork URL if we have it

    # 3. Update Firestore
    db.collection(collection_name).document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "url": audio_url}

# --- Celery Task Definitions ---
@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration, voices):
    print(f"WORKER: Started PODCAST job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    audio_filepath = f"{job_id}.mp3"

    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP, 'voices': voices})
        
        # 1. Generate the script
        original_script = generate_script_from_idea(topic, context, duration)
        
        # 2. Generate the artwork URL
        artwork_url = generate_artwork_for_topic(topic)
        
        # 3. Generate the audio
        if not generate_podcast_audio(original_script, audio_filepath, voices): 
            raise Exception("Audio generation failed.")
            
        # 4. Finalize the job
        return _finalize_job(
            job_id, 
            'podcasts', 
            audio_filepath, 
            f"podcasts/{audio_filepath}", 
            generated_script=original_script, 
            artwork_url=artwork_url
        )
    except Exception as e:
        print(f"ERROR in podcast task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(audio_filepath): os.remove(audio_filepath)
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
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
    
@app.route("/debug-env")
def debug_env():
    # This will collect all environment variables the app can see
    env_vars = {key: value for key, value in os.environ.items()}
    return jsonify(env_vars)

if __name__ == '__main__':
    app.run(debug=True, port=5000)