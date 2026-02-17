import requests
import json
import os
from dotenv import load_dotenv
import torch
from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Step 1: Set your Hugging Face API Key
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")


# Step 2: Load Prompt Template
def load_prompt():
    with open("prompts/basic_prompt.txt", "r") as file:
        return file.read()


# Step 3: Define the function to call HF Inference API
def generate_test_cases(user_story):
    prompt_template = load_prompt()

    # Fill in the user story
    prompt = prompt_template.replace("{user_story}", user_story)

    # Define request payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.5,
            "top_p": 0.9
        }
    }

    # Call Hugging Face Inference API (you can use a model like 'mistralai/mistral-7b-instruct' or 'google/flan-t5-xxl')
    response = requests.post(f"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3", headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}, json=payload)

    # Print and inspect the response
    response_json = response.json()
    
    #print("Response data:", response_json)  # Check the structure

    # Handle both cases (list or dict) for the response structure
    if isinstance(response_json, list):  # If the response is a list
        result = response_json[0].get('generated_text', 'No text generated')
    elif isinstance(response_json, dict):  # If it's a dictionary
        result = response_json.get('generated_text', 'No text generated')
    else:
        result = 'Unexpected response format'

    # Print the final generated test cases
    print("Generated Test Cases:", result)


    # # If the response is a list, access the first item (which is the dictionary)
    # if isinstance(response_json, list):
    #     result = response_json[0].get('generated_text', 'No text generated')
    # else:
    #     result = response_json.get('generated_text', 'No text generated')
    #
    # print("Generated Text:", result)
    #
    # if response.status_code != 200:
    #     raise Exception(f"API call failed: {response.text}")
    #
    # result = response.json()
    # generated_text = result.get("generated_text", "")
    #
    # return generated_text

# Step 4: Run the generator
if __name__ == "__main__":
    # Example user story
    user_story = """
    As a user, I want to reset my password so that I can regain access to my account if I forget my password.
    """

    #print("Generating test cases for the following user story:\n")
    #print(user_story)

    try:
        test_cases = generate_test_cases(user_story)
        #print("\n--- Generated Test Cases ---\n")
        #print(test_cases)
    except Exception as e:
        print(f"Failed to generate test cases: {e}")

app = FastAPI()

os.environ["DISABLE_ACCELERATE_HOOKS"] = "1"


model_name = "Kaar7/HF_FineTunedTestCaseGenerator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
generator = pipeline(
    "text-generation",
    model="Kaar7/HF_FineTunedTestCaseGenerator",  # ✅ official model
    device_map="auto"
)

prompt = "Generate test cases for login feature."
print(generator(prompt, max_new_tokens=200)[0]['generated_text'])

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate_test_cases/")
def generate_test_cases(user_story: str):
    output = generator(user_story, max_length=100, num_return_sequences=1)
    generated_text = output[0]["generated_text"]
    print("Generated Text:", generated_text)
    return {"test_cases": generated_text}



