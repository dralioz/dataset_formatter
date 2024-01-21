from easygoogletranslate import EasyGoogleTranslate
from datasets import load_from_disk
import json
from tqdm import tqdm
import os
import pandas as pd

# Remained at 11501

IM_START_TOKEN = '<|im_start|>'
IM_END_TOKEN = '<|im_end|>'

translator = EasyGoogleTranslate(
    source_language='en',
    target_language='tr',
    timeout=10
)

# Replace '/path/to/directory' with the path to your saved dataset
data = load_from_disk('./data')

df = pd.DataFrame(data['train'])

conversations = df[["conversation"]]

output_data = []
count = 0

file_path = 'lmsys-chat-1m.json'

for index, row in tqdm(conversations.iterrows(), total = conversations.shape[0]):
    print(index)
    if index > 53300:
        try:
            prompt = ''
            input_text = row["conversation"][0]["content"]
            output_text = row["conversation"][1]["content"]
            
            if len(input_text) < 5000 and len(output_text) < 5000:
            
                translated_input = translator.translate(input_text)
                translated_output = translator.translate(output_text)
                
                prompt += IM_START_TOKEN + 'user\n' + translated_input + IM_END_TOKEN
                prompt += IM_START_TOKEN + 'asistant\n' + translated_output + IM_END_TOKEN
            
            else:
                continue
        
        except Exception as ex:
            print(f"Error Occured | {ex}")
            continue
        print('INFO | ' + prompt)
        
        output_data.append(
        {
                'prompt' : prompt
        }
        )
        
        if index % 100 == 0:
            # Check if the file exists and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # Step 1: Read the existing data
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                # If file doesn't exist or is empty, start with an empty list
                data = []

            # Step 2: Append new data to the list
            data.extend(output_data)  # Use extend to add elements of output_data to data

            # Step 3: Write the updated data back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            
            file.close()
            
            # Reset output_data
            output_data = []
    
    else:
        continue

