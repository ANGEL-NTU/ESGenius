# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
from torch.utils.data import DataLoader  # Still needed for DataLoader
from tqdm import tqdm
import requests  # Specific import for Qwen/Dashscope
import traceback

# Ensure current directory is in the Python path
sys.path.append(os.getcwd())

# Import shared components from the utility file
import evaluation_utils as utils

# =====================================================================
# --------- Qwen (Dashscope) API Model Evaluation Functions -----------
# =====================================================================

# --- Helper Function for Qwen API ---
def query_qwen_api(prompt, model_name):
    """Queries Qwen (Dashscope) API. Returns response content or error marker."""
    # Uses DASHSCOPE_API_KEY from utils
    if not utils.DASHSCOPE_API_KEY:
        return "API_KEY_MISSING"

    # Map user-facing model names to API identifiers if necessary
    api_model_mapping = {
        "Qwen2.5-Max": "qwen-max-2025-01-25",  # Example: Check Dashscope docs for exact identifiers
        # Add other mappings as needed
    }
    api_model = api_model_mapping.get(model_name, model_name)  # Use mapping or original name

    api_key = utils.DASHSCOPE_API_KEY
    API_ENDPOINT_COMPAT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"  # International

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Use COMMON hyperparameters from utils for generation
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": utils.TEMPERATURE,
        "top_p": utils.TOP_P,
        "n": 1
    }
    try:
        response = requests.post(API_ENDPOINT_COMPAT, json=payload, headers=headers, timeout=30)  # Increased timeout
        # Check status codes carefully
        if response.status_code == 429:
            return "API_RATE_LIMIT"
        if response.status_code == 401:
            print(f"Qwen Auth Error for {model_name} ({api_model}). Check API Key.")
            return "API_KEY_INVALID"
        # Check for other specific Dashscope errors if known
        response.raise_for_status()  # Raise HTTPError for other bad responses (4xx or 5xx)
        data = response.json()
        # Parse response, checking structure
        if "choices" in data and len(data["choices"]) > 0 and "message" in data["choices"][0]:
            content = data["choices"][0]["message"].get("content")
            return content if content else "EMPTY_RESPONSE"  # Return marker if empty
        elif "code" in data:
            print(f"Qwen API Error Response: {data}")
            return f"API_ERROR ({data.get('code')}: {data.get('message', 'No message')})"
        else:
            print(f"Unexpected Qwen response format: {data}")
            return "API_ERROR (UnknownFormat)"
    except requests.exceptions.Timeout:
        print(f"Timeout Qwen API ({model_name})")
        return "API_TIMEOUT"
    except requests.exceptions.RequestException as e:
        print(f"Network Error Qwen API ({model_name}): {type(e).__name__}")
        return "API_ERROR"  # Network errors
    except Exception as e:
        print(f"Unexpected Qwen error ({model_name}): {type(e).__name__} - {e}")
        return "API_ERROR"

def evaluate_qwen_api(df, model_name, dataset_name, eval_df, evaluation_excel_file):
    """Function to evaluate Qwen API models with batch saving."""
    max_retries = 10
    backoff_factor = 2
    error_markers = {"API_RATE_LIMIT", "API_ERROR", "API_KEY_MISSING", "API_KEY_INVALID", "API_TIMEOUT", "EMPTY_RESPONSE"}
    # Include specific Qwen errors if identified
    qwen_specific_errors = {"API_ERROR (InvalidParameter: ...)", "API_ERROR (UnknownFormat)"}
    error_markers.update(qwen_specific_errors)
    retryable_errors = {"API_RATE_LIMIT", "API_TIMEOUT", "API_ERROR"}  # Basic retry set

    dataset = utils.QADataset(df, dataset_name=dataset_name)
    effective_batch_size = min(utils.BATCH_SIZE_API, len(dataset))
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=utils.NUM_WORKERS_API)
    pbar = tqdm(total=len(df), desc=f"API Eval (Qwen): {model_name}", unit="q")

    for batch_idx, (query_ids, prompts, answers) in enumerate(dataloader):
        batch_results_recorded = False
        for qid, prompt, true_answer in zip(query_ids, prompts, answers):
            generated_text_final = ""
            final_pred = utils.INVALID_ANSWER_MARKER
            for attempt in range(max_retries):
                # print('qid', qid)
                raw_response = query_qwen_api(prompt, model_name)
                # print('raw_response', raw_response)
                if raw_response is None:
                    raw_response = "EMPTY_RESPONSE"

                # Check against expanded error markers
                is_error = (raw_response in error_markers) or any(marker in raw_response for marker in error_markers if isinstance(marker, str))
                if not is_error:  # Successful response
                    generated_text_final = raw_response
                    # print('generated_text_final', generated_text_final)
                    validated_pred = utils.validate_prediction(generated_text_final)
                    #print('validated_pred', validated_pred)
                    final_pred = validated_pred
                    break
                else:
                    generated_text_final = raw_response

                # Retry logic (check against basic retryable set)
                if generated_text_final in retryable_errors and attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
                else:
                    if generated_text_final == "EMPTY_RESPONSE":
                        final_pred = utils.INVALID_ANSWER_MARKER
                    break  # Exit retry loop

            eval_df = utils.record_result(eval_df, qid, model_name, generated_text_final, final_pred)
            batch_results_recorded = True
            pbar.update(1)

        if batch_results_recorded:
            try:
                # Save using the new two-sheet Excel structure
                utils.save_eval_df_to_excel(eval_df.copy(), evaluation_excel_file, model_name)
            except Exception as batch_save_error:
                print(f"\nERROR saving after Qwen API batch {batch_idx + 1}: {batch_save_error}")
    pbar.close()
    return eval_df

# =====================================================================
# -------------------- Main Execution Logic ---------------------------
# =====================================================================

def main():
    """Main function for evaluating Qwen (Dashscope) API models."""
    script_start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
    print(f"--- Starting Qwen API Evaluation Script [{current_time}] ---")

    utils.set_random_seeds(utils.SEED)
    utils.load_model_info()


    filename = 'until_id_83407_8868q.csv'
    df, dataset_name = utils.load_dataset(filename=filename)
    if df is None:
        return

    # --- Define ONLY Qwen API Models ---
    qwen_models = [
        'Qwen2.5-Max',  # Will be mapped to an API identifier in query_qwen_api if needed
        # Add other Qwen models like 'qwen-plus', 'qwen-max-longcontext' etc.
    ]
    if not qwen_models:
        print("No Qwen models defined for evaluation. Exiting.")
        return
    print(f"\nQwen API Models scheduled ({len(qwen_models)} total): {qwen_models}")

    # --- Evaluate Models ---
    models_evaluated_count = 0
    for model_index, model_name in enumerate(qwen_models):
        print(f"\n--- Checking Qwen API Model {model_index+1}/{len(qwen_models)}: {model_name} ---")
        model_start_time = time.time()

        # Skip evaluation if dedicated results file already exists for this model
        if utils.check_if_skip_model(model_name, dataset_name, results_folder="results"):
            continue

        models_evaluated_count += 1
        print(f"Evaluating '{model_name}' (Qwen API)...")
        # Load or initialize per-model evaluation DataFrame and Excel file
        eval_df, evaluation_excel_file = utils.load_or_initialize_eval_df(df, dataset_name, model_name, results_folder="results")
        if eval_df is None:
            continue

        try:
            eval_df = evaluate_qwen_api(df, model_name, dataset_name, eval_df, evaluation_excel_file)
            utils.report_model_accuracy(model_name, eval_df)
        except Exception as model_eval_error:
            print(f"\nCRITICAL ERROR during Qwen evaluation call for {model_name}: {model_eval_error}")
            traceback.print_exc()
            raw_col, ans_col = f"{model_name}_raw_res", f"{model_name}_ans"
            first_col = eval_df.columns[0]
            accuracy_row_mask = eval_df[first_col].astype(str).str.strip().str.lower() == "accuracy"
            if raw_col in eval_df.columns:
                eval_df.loc[~accuracy_row_mask, raw_col] = "EVAL_CALL_ERROR"
            if ans_col in eval_df.columns:
                eval_df.loc[~accuracy_row_mask, ans_col] = utils.INVALID_ANSWER_MARKER
            utils.save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)

        model_end_time = time.time()
        print(f"--- Finished processing Qwen model: {model_name} (Duration: {model_end_time - model_start_time:.2f} seconds) ---")

    script_end_time = time.time()
    print("\n--- Qwen API Evaluation Script Finished ---")
    if models_evaluated_count == 0:
        print("No new Qwen models were evaluated.")
    else:
        print(f"{models_evaluated_count} Qwen model(s) evaluated.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")
    print(f"Results file updated: {evaluation_excel_file}")

if __name__ == '__main__':
    main()