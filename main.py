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
from PIL import Image, ImageDraw, ImageFont

print('initializing..')
AUDIO_PATH = '/audio/'
SLIDE_PATH = '/slides/'
GAIN_FACTOR = 3
# first argument is the txt file with the transcript
topic = sys.argv[1]

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
    black text on white background
    """
    slide_path = os.getcwd() + SLIDE_PATH

    if not os.path.exists(slide_path):
        os.makedirs(slide_path)
    
    print('Generating slides...')
    for slide in slides_JSON['slides']:
        title = slide['title']
        slide_num = slide['slide']

        FONT_SIZE = 60
        SLIDE_HEIGHT = 600
        SLIDE_WIDTH = 800

        # create an image with the title
        img = Image.new('RGB', (SLIDE_WIDTH, SLIDE_WIDTH), color = 'white')

        # draw big text in the middle
        fnt = ImageFont.truetype('./fonts/times new roman bold.ttf', FONT_SIZE)
        
        d = ImageDraw.Draw(img)
        

        text_width, text_height = d.textlength(title, font=fnt), FONT_SIZE

        # if the title is too long, cut it
        cut_title = ""
        if text_width > SLIDE_WIDTH:
            while text_width > SLIDE_WIDTH:
                # add last word to the cut_title
                last_word = title.split()[-1]
                title = title[:-len(last_word)]
                cut_title = last_word + cut_title

                text_width, text_height = d.textlength(title, font=fnt), FONT_SIZE
        


        d.text(((SLIDE_WIDTH - text_width)/2,(SLIDE_HEIGHT-text_height)/2), title, font=fnt, fill='black')
        d.text(((SLIDE_WIDTH - text_width)/2,(SLIDE_HEIGHT+text_height)/2 +FONT_SIZE+10), cut_title, font=fnt, fill='black')
        
        img.save(slide_path+ 'slide_' + str(slide_num) + '.png')

    print('All slides have been generated')

def generate_transcript(topic):
    prompt = f"""
You are an experienced programmer and presenter. You are giving a presentation about {topic}.
You have to generate a transcript for the presentation. The transcript should be clear and concise.
Assume your audience is highschool students. Do not exceed 10 slides but it can be lower. 
Each slide must have at least 8 sentences. Make the presentation style engaging by asking rhetorical questions and using humor.
If you use humor, keep it mind it will be read by a text-to-speech engine.
it should be in the following format:

Slide 1:
<things to say>

Slide 2:
<things to say>
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
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(endpoint, json=data, headers=headers)
    response_data = response.json()
    transcript = response_data['choices'][0]['message']['content']

    return transcript


def main():
    print('starting')

    print(f'Generating transcript for presentation about {topic}...')

    transcript = generate_transcript(topic)

    slides_JSON = get_JSON_from_transcript(transcript)

    generate_slides(slides_JSON)


    covert_trascript_to_audio(slides_JSON)

    exit(0)


if __name__ == '__main__':
    main()
