from llm.openai import OpenAIChat
from utils.prompter import Prompter
import os
import json
import datetime
import argparse
import logging
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process dataset and save GPT-4 responses.")
parser.add_argument("--input", default= "input_file.jsonl", type=str, help="Path to the input dataset file.")
parser.add_argument("--output", default= "output_file", type=str, help="Path to the output file where GPT-4 responses will be saved.")
parser.add_argument("--prompter", default= "gpt4", type=str, help="Specify the prompter to use.")
parser.add_argument("--column_name", default= "question", type=str, help="Specify the column_name to use for prompter.")
parser.add_argument("--type", default="", type=str, help="Specify the type")


args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Check if the directory exists, if not, create it
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Derive the paths for output_files.jsonl and metadata.json from the base filename or directory
output_file_path = f"{args.output}/output_files.jsonl"
metadata_file_path = f"{args.output}/metadata.json"


# Load data from file
with open(args.input, "r") as file:
    data = [json.loads(line) for line in file]

# Define llm
llm = OpenAIChat()

prompter = Prompter(args.prompter)

system_message = prompter.get_system_message()


# Send requests and collect responses
responses = []
total_prompt_tokens = 0
total_completion_tokens = 0

error_file_path = f"{args.output}/error_data.jsonl"


MAX_RETRIES = 3  # Set this to the number of retries you want

break_outer_loop = False  # Flag to check if we need to break the outer loop

for entry in tqdm(data, total=len(data), desc="Processing data"):
    if break_outer_loop:
        break
        
    retries = 0
    while retries < MAX_RETRIES:
        try:
            user_message = prompter.get_user_message(entry[args.column_name])
            completion = llm.get_response(system_message, user_message)
            response = completion.choices[0].message.content
            
            total_prompt_tokens += completion.usage.prompt_tokens
            total_completion_tokens += completion.usage.completion_tokens
            responses.append(response)
            
            # If processing was successful, break out of the retry loop
            break
        
        except Exception as e:
            retries += 1
            logging.error(f"An error occurred during processing: {str(e)}. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(10)
            
            # If reached max retries, set the flag to break the outer loop
            if retries == MAX_RETRIES:
                logging.error(f"Max retries reached for entry: {entry[args.column_name]}. Stopping entire processing.")
                break_outer_loop = True
                break


# Save responses to a file
with open(output_file_path, "w") as file:
    for entry, response in zip(data[:len(responses)], responses):
        if args.type == "":
            output_data = {
                    "instruction": entry[args.column_name],
                    "output": response,
                    "input" : ""
                }
        else:
            output_data = {
                    "instruction": entry[args.column_name],
                    "output": response,
                    "input" : "",
                    "type" : entry[args.type]
                }
        file.write(json.dumps(output_data, ensure_ascii=False) + "\n")

logging.info(f"All responses saved to {output_file_path}")



# Define token prices
PROMPT_TOKEN_PRICE = 0.03 / 1000  # Price per token for prompt
COMPLETION_TOKEN_PRICE = 0.06 / 1000  # Price per token for completion

# Calculate total cost
total_prompt_cost = total_prompt_tokens * PROMPT_TOKEN_PRICE
total_completion_cost = total_completion_tokens * COMPLETION_TOKEN_PRICE
total_cost = total_prompt_cost + total_completion_cost


logging.info(f"Total prompt tokens: {total_prompt_tokens}")
logging.info(f"Total completion tokens: {total_completion_tokens}")
logging.info(f"Total prompt cost: ${total_prompt_cost:.2f}")
logging.info(f"Total completion cost: ${total_completion_cost:.2f}")
logging.info(f"Total cost: ${total_cost:.2f}")


# Define the structure of output_data
output_data_structure = {
    "keys": ["instruction", "output", "input"],
    "description": {
        "instruction": "The instruction provided to the model.",
        "output": "The response from the model.",
        "input" : ""
    }
}

# After processing all the data
metadata = {
    "description": prompter.get_description(),
    "total_entries_processed": len(data),
    "total_prompt_tokens": total_prompt_tokens,
    "total_completion_tokens": total_completion_tokens,
    "total_prompt_cost": total_prompt_cost,
    "total_completion_cost": total_completion_cost,
    "total_cost": total_cost,
    "execution_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "model_version": "GPT-4",
    "input_file": args.input,
    "output_file": args.output,
    "output_data_structure": output_data_structure
}

# Save metadata to a file
with open(metadata_file_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

logging.info(f"Metadata saved to {metadata_file_path}")