import base64
import json
import os

from openai import OpenAI
import numpy as np

from aura.dataset.provider import DatasetProvider

SYS_PROMPT = "Imagine a short description of an art piece that would represent this person as a scene, with objects, weather, and motion. Respond with only the description inside quotation marks and ignore the fact that the image is in black and white. Have the description focus on the physical appearance of the art piece, letting us imagine what it looks like solely based on the description"
client = OpenAI()

def image_to_base64(image: np.ndarray):
    """
    Convert an image file to a Base64-encoded string.
    Args:
        image (np.ndarray): n-dimensional nparray
    Returns:
        str: Base64-encoded string of the image.
    """
    arr_bytes = image.tobytes()
    base64_str = base64.b64encode(arr_bytes).decode('utf-8')
    return base64_str

def call_openai_with_image(image_path, system_prompt):
    """
    Call the OpenAI API with an image input and system prompt.
    Args:
        image_path (str): Path to the image file.
        system_prompt (str): System-level instructions for the model.
    Returns:
        str: Model's text response.
    """
    # Convert image to Base64
    image = image_to_base64(image_path)

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": SYS_PROMPT
                }]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{image}"
                },
                },
            ],
            }
        ],
    )

    return response.choices[0].message

def generate_files(images):
    BATCH_SIZE = 5_000

    tasks = []
    metadata = []

    for (image, _, emotion), index in enumerate(images):
        image = image_to_base64(image)
        tasks.append({
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "messages" : [
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": SYS_PROMPT
                        }]
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{image}"
                        },
                        },
                    ],
                    }
                ],
            }
        })

        metadata.append({
            "id": index,
            "emotion": emotion,
            "image": image,
        })

    print("Finshed creating tasks!")
    print(f"Length: {len(tasks)}")
    print("=" * 10)
    
    chunk_count = 1
    for i in range(0, len(tasks), BATCH_SIZE):
        task_chunk = tasks[i:i + BATCH_SIZE]
        tasks_file = f"output/tasks/aura_caption_tasks_{chunk_count}.jsonl"
        with open(tasks_file, 'w') as file:
            for obj in task_chunk:
                file.write(json.dumps(obj) + '\n')
        chunk_count += 1

    print("Finished writting tasks to .jsonl!")
    print("=" * 10)

    with open("output/metadata.jsonl", 'w') as file:
        for obj in metadata:
            file.write(json.dumps(obj) + '\n')
    print("Finished writting metadata to .jsonl!")
    print("=" * 10)

if __name__ == '__main__':
    provider = DatasetProvider("/disk/onyx-scratch/dullo009-fall2024")
    print("Finished downloading dataset!")
    images = provider.train.extend(provider.test)
    print(f"Length: {len(images)}")
    print("=" * 10)

    function = input("Note: You can not submit a batch job if you have not already created your task jsonl file.\nAre you generating task / metadata json files or submitted a batch job (generate / batch / status)? ")
    if function == "generate":
        generate_files(provider.image_paths)
    elif function == "batch":
        print("Starting to submit tasks...")
        print("Here are the submitted batch ids...\n")
        batches = 0
        for f in os.listdir("output/tasks"):
            task_path = os.path.join("output/tasks", f)
            if not os.path.isfile(task_path):
                continue
            
            batches += 1
            batch_file = client.files.create(
                file=open(task_path, "rb"),
                purpose="batch"
            )

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(batch_job.id)

        print(f"{batches} Batches submitted!")
        print("=" * 10)
    elif function == "status":
        print("Printing batches...")
        jobs = list(client.batches.list())
        for batch_job in jobs[::-1]:
            print(f"\n{batch_job}\n")