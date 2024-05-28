import os
import openai
from openai import ChatCompletion
import json, csv
from tqdm import tqdm
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

# ----------------
# global 
openai.api_key = os.environ["OPENAI_API_KEY"]
source_csv_dir = "forgpt_blipopt67b_audiocaps_train2.csv"
# raw_output_csv_dir = "gpt35gen_audiocaps_train_0726.csv"
raw_output_csv_dir = "gpt4gen_audiocaps_train_0913_30000_40000.csv"
# re_output_csv_dir = "re_gpt35gen_audiocaps_train2.csv"
# gpt_model = "gpt-3.5-turbo"
gpt_model = "gpt-4"
# ----------------

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

rows= [] 
with open(source_csv_dir, "r", encoding="utf-8") as f:
    for iline, line in enumerate(f):
        rows.append(json.loads(line))

for row in tqdm(rows):
    image_descriptions = row["captions"]
    labels = row["labels"]

    for idx, i in enumerate(image_descriptions):
        
        flag = True
        while(flag):
            # try:
            response = chat_completion_with_backoff(
                model=gpt_model,
                messages=[
                {"role": "system", "content": 'Generate a simple description of the sound from an image description and sound labels. Here are the rules you should follow:'},
                {"role": "system", "content": '1. Consider the sound labels more than the image description because the labels represent the sound.'},
                {"role": "system", "content": '2. Remove the name of the place, city, or country.'},
                {"role": "system", "content": '3. Delete the time, device names, proper name modifiers, number modifiers, and unit modifiers.'},
                {"role": "system", "content": '4. Do not use expressions such as "the sound of ~" or "the audio of ~" or "you can hear ~" because it is obvious.'},
                {"role": "system", "content": '5. Generate 3 descriptions.'},
                {"role": "system", "content": '6. Each description will always consist of one sentence.'},
                {"role": "system", "content": '7. Eliminate information that can only be obtained from visual information contained in the image description, such as the color and shape of clothing and objects.'},
                {"role": "user", "content": f'Image description : "{i}" Sound label : {labels}'}
                ],
            )
            flag=False
            # time.sleep(5)
            # except:
                # print("Some error happened here.")
        # print(response.choices[0]["message"]["content"].strip().split("\n"))
        
        try:
            captions = response.choices[0]["message"]["content"].strip().split("\n")
        except:
            captions = None
            with open("/home/imotolab/data/work/tsubaki/at_pair_generation/gpt_api/error_data.txt", "a") as file:
                    file.write("Error caption: " + i + "\n")
                
        raw_out_d= {f"raw_gpt_caption_{idx}" : captions}
        row.update(raw_out_d)
        
    with open(raw_output_csv_dir, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([json.dumps(row)])