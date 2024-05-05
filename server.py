from flask import Flask, request, Response
import requests
from google.cloud import speech, language_v1
from google.oauth2 import service_account
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import firebase_admin
from firebase_admin import credentials, firestore
from twilio.twiml.voice_response import VoiceResponse, Gather, Record
from datetime import datetime
import io
import time
import random
import string
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import logging
import pandas as pd
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def summarize_text_bart(transcript):
    inputs = bart_tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_pipe = pipeline("ner", model=ner_model, tokenizer=tokenizer)
roberta_model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
roberta_tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
google_creds = service_account.Credentials.from_service_account_file('')
speech_client = speech.SpeechClient(credentials=google_creds)
language_client = language_v1.LanguageServiceClient(credentials=google_creds)

firebase_cred = credentials.Certificate('')
firebase_admin.initialize_app(firebase_cred)
db = firestore.client()

@app.route("/incoming_call", methods=['POST'])
def incoming_call():
    resp = VoiceResponse()
    gather = Gather(num_digits=1, action='/handle_key_press', method='POST')
    gather.say("Hey, all operators are busy right now. If you want, you can wait, or talk to me instead, and I'll process the emergency. Press 0 if you want to talk now.")
    resp.append(gather)
    resp.redirect('/incoming_call')
    return str(resp)

@app.route("/handle_key_press", methods=['POST'])
def handle_key_press():
    digit_pressed = request.form['Digits']
    resp = VoiceResponse()
    if digit_pressed == '0':
        resp.say("Okay. Please describe your emergency after the beep. Press # when finished.")
        resp.record(timeout=10, finishOnKey='#', action='/handle_recording', maxLength=3600, playBeep=True)
    else:
        resp.say("Please wait while we connect you to an operator.")
    return str(resp)

@app.route("/handle_recording", methods=['POST'])
def handle_recording():
    recording_url = request.values.get('RecordingUrl')
    if not recording_url:
        return Response("No recording URL provided", status=400)
    try:
        download_recording(recording_url, ('', ''))
        transcription = transcribe_audio()
        locations = extract_location_roberta(transcription)  # Use the new RoBERTa-based function
        summary = summarize_text(transcription)
        save_to_firebase({'transcription': transcription, 'summary': summary, 'locations': locations})
        assign_group_identifiers(transcription)
        return Response("Processed successfully", status=200)
    except Exception as e:
        logging.error(f"Error processing recording: {str(e)}", exc_info=True)
        return Response(f"Error processing recording: {str(e)}", status=500)

def summarize_text(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    annotations = language_client.analyze_entities(document=document)
    key_sentences = []

    important_entities = {entity.name.lower() for entity in annotations.entities if entity.salience > 0.01}

    for sentence in text.split('. '):
        if any(entity in sentence.lower() for entity in important_entities):
            key_sentences.append(sentence)

    return ' '.join(key_sentences)

def download_recording(url, auth):
    """Download the recording file from Twilio with retry logic and delay."""
    attempts = 0
    while attempts < 5:  # Retry up to 5 times
        response = requests.get(url, auth=auth, stream=True)
        if response.status_code == 200:
            with open("output.wav", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed successfully. File saved as 'output.wav'.")
            return
        else:
            print(f"Attempt {attempts+1}: Failed to download recording. Status: {response.status_code}. Retrying in 5 seconds...")
            time.sleep(5)  
            attempts += 1
    print("Failed to download recording after multiple attempts.")

def transcribe_audio():
    """Transcribe audio using Google Speech-to-Text directly from a WAV file."""
    with io.open('output.wav', 'rb') as audio_file:
        audio_content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000, #twillio api sample rate i think
        language_code="en-US"
    )
    response = speech_client.recognize(config=config, audio=audio)
    return ' '.join(result.alternatives[0].transcript for result in response.results)

def save_to_firebase(data):
    current_time = datetime.now().isoformat()
    data.update({
        'timestamp': current_time,
        'Group': ''
    })
    doc_ref = db.collection('transcriptions').document()
    doc_ref.set(data)

def extract_location_roberta(transcript):
    inputs = roberta_tokenizer.encode_plus("Where did the incident occur? " + transcript, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = roberta_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = roberta_tokenizer.convert_tokens_to_string(roberta_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def fetch_all_transcripts():
    docs = db.collection('transcriptions').stream()
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data['id'] = doc.id 
        data.append(doc_data)
    return data

def assign_group_identifiers(new_transcript):
    existing_data = fetch_all_transcripts()
    introductions_df = pd.DataFrame(existing_data)
    if not introductions_df.empty and 'transcription' in introductions_df.columns:
        user_messages = introductions_df['transcription'].tolist()
        message_embeddings = bert_model.encode(user_messages, convert_to_tensor=True)
        new_transcript_embedding = bert_model.encode([new_transcript], convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(new_transcript_embedding, message_embeddings)[0].cpu().numpy()

        threshold = 0.6  #adjusted again ffs
        for index, similarity in enumerate(similarities):
            print(f"Similarity with index {index}: {similarity}")
            if similarity > threshold:
                group_id = introductions_df.iloc[index]['Group']
                if group_id == '':
                    group_id = ''.join(random.choices(string.ascii_letters + string.digits, k=26))
                    db.collection('transcriptions').document(introductions_df.iloc[index]['id']).update({'Group': group_id})
                new_id = introductions_df[introductions_df['transcription'] == new_transcript]['id'].values[0]
                db.collection('transcriptions').document(new_id).update({'Group': group_id})

sample_data = [ #????
    {
        'transcription': 'There is a fire in the central market area. It is spreading fast.',
        'summary': 'Fire in central market area, spreading fast.',
        'locations': 'central market area',
        'timestamp': '2023-05-01T12:00:00',
        'Group': ''
    },
    {
        'transcription': 'There is a fire in the central market area. It is spreading fast.',
        'summary': 'Fire in central market area, spreading fast.',
        'locations': 'central market area',
        'timestamp': '2023-05-01T12:05:00',
        'Group': ''
    },
    {
        'transcription': 'Multiple car collision in the central market area. Many injured.',
        'summary': 'Car collision in central market, many injured.',
        'locations': 'central market area',
        'timestamp': '2023-05-01T12:10:00',
        'Group': ''
    },
    {
        'transcription': 'A robbery at 5th street bank, suspects are armed.',
        'summary': 'Robbery at 5th street bank, suspects armed.',
        'locations': '5th street bank',
        'timestamp': '2023-05-01T12:15:00',
        'Group': ''
    },
    {
        'transcription': 'Suspicious activity reported near the old warehouse on 22nd street.',
        'summary': 'Suspicious activity near old warehouse on 22nd.',
        'locations': 'old warehouse on 22nd street',
        'timestamp': '2023-05-01T12:20:00',
        'Group': ''
    },
    {
        'transcription': 'Someone fell into the river at Riverside Park, needs immediate rescue.',
        'summary': 'Person fell into river at Riverside Park, needs rescue.',
        'locations': 'Riverside Park',
        'timestamp': '2023-05-01T12:25:00',
        'Group': ''
    }
]

def upload_initial_data():
    for record in sample_data:
        doc_ref = db.collection('transcriptions').document()
        doc_ref.set(record)

if __name__ == "__main__":
    try:
        """upload_initial_data()  # Preload data into Firebase
        assign_group_identifiers("There is a fire in the central market area. It is spreading fast.")"""
        app.run(debug=True, port=8000)
    except Exception as e:
        logging.error(f"Failed to upload initial data or start the app: {str(e)}")
