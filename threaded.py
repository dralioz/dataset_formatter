import concurrent.futures
from easygoogletranslate import EasyGoogleTranslate

from datasets import load_from_disk
import json
from tqdm import tqdm
import os
import pandas as pd

import time
from datetime import datetime

IM_START_TOKEN = '<|im_start|>'
IM_END_TOKEN = '<|im_end|>'

translator = EasyGoogleTranslate(
    source_language='en',
    target_language='tr',
    timeout = 10
)

def translate_texts(input_text, output_text):
    # Check if the input text length is less than or equal to 5000 characters
    if len(input_text) <= 5000 and len(output_text) <= 5000 :
        try:
            inp_text = translator.translate(input_text)
            out_text = translator.translate(output_text)
            return inp_text,out_text
        except Exception as ex:
            with open('error.txt', 'a') as f:
                f.write(str(ex) + '\n')
            f.close()
            return None, None
    else:
        # Return None or appropriate values for texts that are too long
        return None, None


def main(start_index, end_index, frame):

    input_texts = frame["input_text"][start_index:end_index].to_list()
    output_texts = frame["output_text"][start_index:end_index].to_list()

    start_time = time.time()
    print(f"Translate is started at {datetime.now()} ")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use zip to pair each input_text with its corresponding output_text
        results = list(executor.map(lambda pair: translate_texts(*pair), zip(input_texts, output_texts)))

        # Unzipping the results into two separate lists
        inputs, outputs = zip(*results)

        # Optionally, filter out the None values if they are not needed
        inputs = [input for input in inputs if input is not None]
        outputs = [output for output in outputs if output is not None]
    # Replace with your actual end token

    end_time = time.time()
    # print(f"Translate is finished at {datetime.now()}. Total duration is {end_time - start_time} seconds.")
    
    return inputs, outputs
    

data = load_from_disk('./data')
df = pd.DataFrame(data['train'])

conversations = df[["conversation"]]

frame = pd.DataFrame()

frame['input_text'] = conversations['conversation'].apply(lambda x: x[0]['content'] if x[0]['role'] == 'user' else x[1]['content'])
frame['output_text'] = conversations['conversation'].apply(lambda x: x[1]['content'] if x[1]['role'] == 'assistant' else x[0]['content'])

data = []
start_index = 183300
start_time = time.time()
file_count = 131

for i in range(1600):
    time.sleep(2)
    end_index = start_index + 50
    inputs, outputs = main(start_index, end_index, frame)
    prompts = []
    for input_text, output_text in zip(inputs, outputs):
        prompt = ""
        prompt += IM_START_TOKEN + 'user\n' + input_text + IM_END_TOKEN
        prompt += IM_START_TOKEN + 'assistant\n' + output_text + IM_END_TOKEN
        prompts.append(
            {
                'prompt' : prompt
            }
        )
        # print('INFO | ' + prompt)
    data.extend(prompts)
    start_index = end_index
    
    if (i % 20 == 0 and i != 0) or i == 1599:
        print(20*50, i, len(prompts), len(data))
        time.sleep(220)
        
        file_path = f'lmsys-chat-1m_21_Jan_1000_{file_count}.json'
        file_count += 1
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        file.close()
        data = []

end_time = time.time()
print(f"Prompt adjustment is finished at {datetime.now()}. Total duration is {end_time - start_time} seconds.")