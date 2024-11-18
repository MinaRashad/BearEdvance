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
import PIL

print('initializing..')
AUDIO_PATH = '/audio/'
SLIDE_PATH = '/slides/'
GAIN_FACTOR = 3
# first argument is the txt file with the transcript
transcript = sys.argv[1]

transcript_str = ''

try:
    with open(transcript, 'r') as file:
        transcript_str = file.read()
except Exception as e:
    print('Error reading file: ', e)



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

    # cut text into sentences
    text_pieces = []

    # find number of sentences
    num_sentences = len(re.findall(r'[.!?]', text))
    while len(text) > 0:
        # look for . or ? or ! to cut the text
        next_dot = text.find(".")
        next_question = text.find("?")
        next_exclamation = text.find("!")
        
        endings = []
        for end in [next_dot, next_question, next_exclamation]:
            if end != -1:
                endings.append(end)
        
        if len(endings) == 0:
            # no more sentence endings, add the rest of the text
            text_pieces.append(text)
            break
        
        end = min(endings)+1

        # get the sentence
        sentence = text[:end].strip()

        # remove the sentence from the text
        text = text[end:]

        text_pieces.append(sentence)

        

    
    
    for i, piece in enumerate(text_pieces):
        
        # print the progress
        print(f'Processing sentence {i+1}/{num_sentences}', end='\r')

        speech = synthesiser(piece, forward_params={"speaker_embeddings": speaker_embedding})
        increased_volume_audio = np.clip(speech["audio"] * GAIN_FACTOR, -1.0, 1.0)
        sf.write(file_path + "_part_" + str(i) + ".wav", increased_volume_audio, samplerate=speech["sampling_rate"])

def get_JSON_from_transcript(transcript):
    print('Using OpenAI to convert the transcript to a JSON format...')
    prompt = """
You are an experienced programmer and presenter. You have the following transcript to present. 
make sure to change abbreviations to full words. Dont use any punctuation.
Convert them into JSON with the following format:
{
slides:[
    { 
    slide: 1,
    line: <Things to say>,
    title: <summary of things to say in 5 words at most>
    },
    {
    slide: <slide number>,
    line: <Things to say>,
    title: <summary of things to say in 5 words at most>
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
    slides_JSON = json.loads(slides_JSON)

    return slides_JSON

def covert_trascript_to_audio(slides_JSON):
    print('Transcript converted to JSON, starting to convert to audio...')
    for slide in slides_JSON['slides']:
        mytext = slide['line']
        print('Converting slide ' + str(slide['slide']) + ' to audio...')
        speak(mytext, "slide" + str(slide['slide']))

    print("All slides have been converted to audio")

def generate_slides(slides_JSON):
    """
    using the title from the JSON, we will generate an image with that title
    black monospace text on white background
    """
    slide_path = os.getcwd() + SLIDE_PATH

    if not os.path.exists(slide_path):
        os.makedirs(slide_path)
    
    print('Generating slides...')
    for slide in slides_JSON['slides']:
        title = slide['title']
        slide_num = slide['slide']

        # create an image with the title
        img = PIL.Image.new('RGB', (800, 600), color = 'white')

        fnt = PIL.ImageFont.load_default()
        d = PIL.ImageDraw.Draw(img)
        d.text((10,10), title, font=fnt, fill=(0,0,0))

        img.save(slide_path+ 'slide_' + str(slide_num) + '.png')

    print('All slides have been generated')


def main():
    print('starting')

    slides_JSON = get_JSON_from_transcript(transcript_str)

    covert_trascript_to_audio(slides_JSON)

    generate_slides(slides_JSON)

    exit(0)


