# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import traceback

# Ensure current directory is in the Python path
sys.path.append(os.getcwd())

# Import shared components from the utility file
import evaluation_utils as utils

# =====================================================================
# ------ Open Source Model Evaluation Function (Uses Local Compute) ---
# =====================================================================

def evaluate_open_source_model(df, model_name, dataset_name, eval_df, evaluation_excel_file):
    """Evaluates a single local Hugging Face model with batch saving."""
    max_retries = 10
    model, tokenizer, generator = None, None, None
    model_dtype = torch.float16  # Or bfloat16 if supported and preferred

    try:
        print(f"Loading Model & Tokenizer: {model_name}")
        # --- Loading ---
        tokenizer_args = {"trust_remote_code": True, "use_fast": True, "token": utils.HF_TOKEN}
        model_args = {"trust_remote_code": True, "device_map": "auto", "torch_dtype": model_dtype, "token": utils.HF_TOKEN}

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)

        if "google/gemma-3" in model_name:
            try:
                from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
                model = Gemma3ForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    device_map="auto",
                    torch_dtype=torch.bfloat16, 
                    token=utils.HF_TOKEN
                )
            except ImportError:
                print("Gemma3 specific class not found, using AutoModelForCausalLM.")
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        elif "Qwen/Qwen2.5-Omni" in model_name:
            from transformers import AutoModelForSeq2SeqLM

            # Load the model with specified configurations
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
            
        # --- Pad Token Handling ---
        if tokenizer and not tokenizer.pad_token:
            if tokenizer.eos_token:
                print("Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                print("Adding new [PAD] token.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings

        if not model or not tokenizer:
            raise ValueError("Model or Tokenizer failed to load.")

        # --- Pipeline Setup ---
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            do_sample=False,
            temperature=utils.TEMPERATURE,
            top_p=utils.TOP_P,
            top_k=utils.TOP_K,
            pad_token_id=tokenizer.pad_token_id
        )
        print(f"Pipeline created on device(s): {generator.device}")

        # --- DataLoader ---
        dataset = utils.QADataset(df, dataset_name)
        effective_batch_size = min(utils.BATCH_SIZE_GPU, len(dataset))
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=utils.NUM_WORKERS_GPU, pin_memory=torch.cuda.is_available())

        pbar = tqdm(total=len(df), desc=f"GPU Eval: {model_name}", unit="q")

        # --- Batch Processing ---
        for batch_idx, (query_ids, prompts, answers) in enumerate(dataloader):
            generated_outputs = None
            for attempt in range(max_retries):
                try:
                    prompt_list = list(prompts)
                    outputs = generator(prompt_list)
                    generated_outputs = outputs
                    break
                except torch.cuda.OutOfMemoryError as oom_err:
                    print(f"\nOOM Error: Batch {batch_idx+1}, Attempt {attempt+1}. Trying memory clear.")
                    print(f"Error details: {oom_err}")
                    del outputs
                    utils.cleanup_gpu_resources(generator=None, model=None, tokenizer=None)
                    time.sleep(5 * (attempt + 1))
                    generated_outputs = None
                    if attempt == max_retries - 1:
                        print("OOM Error persisted after retries.")
                except Exception as e:
                    print(f"\nGeneration Error: Batch {batch_idx+1}, Attempt {attempt+1}: {type(e).__name__} - {e}")
                    traceback.print_exc()
                    utils.cleanup_gpu_resources(generator=None, model=None, tokenizer=None)
                    time.sleep(2 ** attempt)
                    generated_outputs = None
                    if attempt == max_retries - 1:
                        print("Generation Error persisted after retries.")

            # --- Process Batch Results ---
            batch_results_recorded = False
            if generated_outputs is not None and isinstance(generated_outputs, list):
                min_len = len(query_ids)
                if len(generated_outputs) != len(query_ids):
                    print(f"\nWarning: Output/Input count mismatch in batch {batch_idx+1}. Processing minimum of {min(len(generated_outputs), len(query_ids))}.")
                    min_len = min(len(generated_outputs), len(query_ids))

                for i in range(min_len):
                    qid = query_ids[i]
                    prompt_used = prompts[i]
                    output_item = generated_outputs[i]
                    raw_output, final_pred = "", utils.INVALID_ANSWER_MARKER

                    try:
                        if isinstance(output_item, list) and len(output_item) > 0 and isinstance(output_item[0], dict):
                            generated_text = output_item[0].get('generated_text', '')
                        elif isinstance(output_item, dict):
                            generated_text = output_item.get('generated_text', '')
                        else:
                            generated_text = str(output_item)
                            print(f"Warning: Unexpected output item format: {output_item}")

                        if generated_text:
                            separator = "\nAnswer:"
                            if generated_text.startswith(prompt_used):
                                raw_output = generated_text[len(prompt_used):].strip()
                            elif separator in generated_text:
                                raw_output = generated_text.split(separator)[-1].strip()
                            else:
                                raw_output = generated_text.strip()
                                if len(raw_output) > 10:
                                     print(f"Warning: Fallback parsing for QID {qid} resulted in long output: '{raw_output[:50]}...'")
                            validated_pred = utils.validate_prediction(raw_output)
                            final_pred = validated_pred
                        else:
                            raw_output = "EMPTY_GENERATION"
                            final_pred = utils.INVALID_ANSWER_MARKER
                    except Exception as parse_error:
                        print(f"\nError parsing output for QID {qid}: {parse_error}")
                        raw_output = f"PARSE_ERROR: {parse_error}"
                        final_pred = utils.INVALID_ANSWER_MARKER

                    eval_df = utils.record_result(eval_df, qid, model_name, raw_output, final_pred)
                    batch_results_recorded = True
                    pbar.update(1)
            else:
                print(f"\nRecording generation failures for batch {batch_idx+1}...")
                for qid_fail in query_ids:
                    eval_df = utils.record_result(eval_df, qid_fail, model_name, "GENERATION_FAIL", utils.INVALID_ANSWER_MARKER)
                    batch_results_recorded = True
                    pbar.update(1)

            if batch_results_recorded:
                try:
                    utils.save_eval_df_to_excel(eval_df.copy(), evaluation_excel_file, model_name)
                except Exception as batch_save_error:
                     print(f"\nERROR saving after GPU batch {batch_idx + 1}: {batch_save_error}")

        pbar.close()

    except (ImportError, OSError, ValueError, torch.cuda.OutOfMemoryError) as load_err:
         print(f"\nFATAL ERROR setting up model {model_name}: {type(load_err).__name__} - {load_err}. Skipping.")
         traceback.print_exc()
         raw_col, ans_col = f"{model_name}_raw_res", f"{model_name}_ans"
         if raw_col not in eval_df.columns: eval_df[raw_col] = pd.NA
         if ans_col not in eval_df.columns: eval_df[ans_col] = pd.NA
         first_col = eval_df.columns[0]
         accuracy_row_mask = eval_df[first_col].astype(str).str.strip().str.lower() == "accuracy"
         eval_df.loc[~accuracy_row_mask, raw_col] = "SETUP_FAIL"
         eval_df.loc[~accuracy_row_mask, ans_col] = utils.INVALID_ANSWER_MARKER
         utils.save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)
    except Exception as e:
         print(f"\nUNEXPECTED FATAL error during run for {model_name}: {type(e).__name__} - {e}")
         traceback.print_exc()
         raw_col, ans_col = f"{model_name}_raw_res", f"{model_name}_ans"
         if raw_col not in eval_df.columns: eval_df[raw_col] = pd.NA
         if ans_col not in eval_df.columns: eval_df[ans_col] = pd.NA
         first_col = eval_df.columns[0]
         accuracy_row_mask = eval_df[first_col].astype(str).str.strip().str.lower() == "accuracy"
         eval_df.loc[~accuracy_row_mask, raw_col] = "UNEXPECTED_ERROR"
         eval_df.loc[~accuracy_row_mask, ans_col] = utils.INVALID_ANSWER_MARKER
         utils.save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)
         
    utils.cleanup_gpu_resources(generator, model, tokenizer)

    return eval_df

# =====================================================================
# -------------------- Main Execution Logic ---------------------------
# =====================================================================

def main():
    """Main function for evaluating open-source models."""
    script_start_time = time.time()
    print(f"--- Starting Open Source Evaluation Script [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---")

    utils.set_random_seeds(utils.SEED)
    utils.perform_hf_login()
    utils.load_model_info()

    # filename = 'IPCC_balanced_288q.csv'
    # filename = 'GRI_balanced_230q.csv'
    filename = 'SASB_balanced_266q.csv'
    df, dataset_name = utils.load_dataset(filename=filename)
    if df is None:
        return

    # Define the list of open-source models to evaluate
    open_source_models = [
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'google/gemma-3-1b-pt',
        'google/gemma-3-1b-it',
        'google/gemma-3-4b-pt',
        'google/gemma-3-4b-it',
        'google/gemma-3-12b-pt',
        'google/gemma-3-12b-it',
        'google/gemma-3-27b-pt',
        'google/gemma-3-27b-it',
        'meta-llama/Meta-Llama-3-8B',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.1-8B',
        'meta-llama/Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.2-1B',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct',
        'Qwen/Qwen2.5-0.5B',
        'Qwen/Qwen2.5-0.5B-Instruct',
        'Qwen/Qwen2.5-1.5B',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'Qwen/Qwen2.5-3B',
        'Qwen/Qwen2.5-3B-Instruct',
        'Qwen/Qwen2.5-7B',
        'Qwen/Qwen2.5-7B-Instruct',
        'Qwen/Qwen2.5-14B',
        'Qwen/Qwen2.5-14B-Instruct',
        'Qwen/Qwen2.5-32B',
        'Qwen/Qwen2.5-32B-Instruct',
        'Qwen/Qwen2.5-72B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct-1M',
        'Qwen/Qwen2.5-14B-Instruct-1M', 
        'Qwen/QwQ-32B',
        # 'Qwen/Qwen2.5-Omni-7B', # cannot work with this model
        'Qwen/Qwen3-0.6B',
        'Qwen/Qwen3-1.7B',
        'Qwen/Qwen3-4B',
        'Qwen/Qwen3-8B',
        # 'Qwen/Qwen3-14B', # download error
        # 'Qwen/Qwen3-32B', # download error
        
    ]
    if not open_source_models:
        print("No open-source models defined for evaluation. Exiting.")
        return
    print(f"\nOpen Source Models scheduled ({len(open_source_models)} total): {open_source_models}")

    models_evaluated_count = 0
    for model_index, model_name in enumerate(open_source_models):
        print(f"\n--- Checking Open Source Model {model_index+1}/{len(open_source_models)}: {model_name} ---")
        model_start_time = time.time()

        # Skip evaluation if a results file already exists for this model
        if utils.check_if_skip_model(model_name, dataset_name, results_folder="results"):
            continue

        models_evaluated_count += 1
        print(f"Evaluating '{model_name}' (Open Source)...")
        # Initialize evaluation DataFrame and dedicated Excel file for the current model
        eval_df, evaluation_excel_file = utils.load_or_initialize_eval_df(df, dataset_name, model_name, results_folder="results")
        if eval_df is None:
            continue

        try:
            eval_df = evaluate_open_source_model(df, model_name, dataset_name, eval_df, evaluation_excel_file)
            utils.report_model_accuracy(model_name, eval_df)
        except Exception as model_eval_error:
            print(f"\nCRITICAL ERROR during evaluation call for model {model_name}: {model_eval_error}")
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
        print(f"--- Finished processing model: {model_name} (Duration: {model_end_time - model_start_time:.2f} seconds) ---")

    script_end_time = time.time()
    print("\n--- Open Source Evaluation Script Finished ---")
    if models_evaluated_count == 0:
        print("No new open-source models were evaluated in this run.")
    else:
        print(f"{models_evaluated_count} open-source model(s) were newly evaluated or re-evaluated.")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds ({((script_end_time - script_start_time)/3600):.2f} hours)")



if __name__ == '__main__':
    main()