import os
import sys
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import numpy as np
import requests
import json
import re

print('initializing..')
AUDIO_PATH = '/audio/'
GAIN_FACTOR = 3
# first argument is the txt file with the transcript
transcript = sys.argv[1]

transcript_str = ''

try:
    with open(transcript, 'r') as file:
        transcript_str = file.read()
except Exception as e:
    print('Error reading file: ', e)

# remove special characters with regex
transcript_str = re.sub(r'[^a-zA-Z0-9\s]', '', transcript_str)

# audio synthesis
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# get the open AI API key by reading config.json
with open('config.json', 'r') as file:
    config = json.load(file)

OPENAI_API_KEY = config['OPENAI_API_KEY']


def speak(text, filename):
    GAIN_FACTOR = 3  # Adjust this value to increase or decrease the volume
    current_path = os.getcwd()
    file_path = current_path + AUDIO_PATH + filename

    # check audio folder
    if not os.path.exists(current_path + AUDIO_PATH):
        os.makedirs(current_path + AUDIO_PATH)

    # cut text into pieces with max length of 100 words
    words = text.split()
    text_pieces = []
    while len(words) > 0:
        piece = " ".join(words[:100])
        text_pieces.append(piece)
        words = words[100:]
    
    for i, piece in enumerate(text_pieces):
        speech = synthesiser(piece, forward_params={"speaker_embeddings": speaker_embedding})
        increased_volume_audio = np.clip(speech["audio"] * GAIN_FACTOR, -1.0, 1.0)
        sf.write(file_path + "_part_" + str(i) + ".wav", increased_volume_audio, samplerate=speech["sampling_rate"])

def get_JSON_from_transcript(transcript):
    prompt = """
You are an experienced programmer and presenter. You have the following transcript to present. 
make sure to change abbreviations to full words.
Convert them into JSON with the following format:
{
slides:[
    { 
    slide: 1,
    line: <Things to say>
    },
    {
    slide: <slide number>,
    line: <Things to say>
    }
]
}

Here is the transcript

"""

    endpoint =  "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_API_KEY,
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    }

    response = requests.post(endpoint, json=data, headers=headers)
    response_data = response.json()
    slides_JSON = response_data['choices'][0]['message']['content']
    slides_JSON = eval(slides_JSON)

    return slides_JSON

def covert_trascript_to_audio(transcript):
    slides_JSON = get_JSON_from_transcript(transcript)

    for slide in slides_JSON['slides']:
        mytext = slide['line']
        speak(mytext, "slide" + str(slide['slide']))

    print("All slides have been converted to audio")

print('starting')
covert_trascript_to_audio(transcript_str)

# create slide content JSON file
def create_slide_content_json(transcript):
    prompt = """

You are an experienced programmer and presenter. You have the following transcript to present.
Convert them into JSON with the following format:
{
    slides:[
        {
            slide: 1,
            title: <summary of things to say in 5 words at most>
        },
        {
            slide: <slide number>,
            line: <slide title>
        }
    ]
    }
    
    Here is the transcript:
    """

    endpoint = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_API_KEY,
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    }

    response = requests.post(endpoint, json=data, headers=headers)
    response_data = response.json()
    slides_JSON = response_data['choices'][0]['message']['content']
    slides_JSON = eval(slides_JSON)

    return slides_JSON

# we need to create images with the slide summary



