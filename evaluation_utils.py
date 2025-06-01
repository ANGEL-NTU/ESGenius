# -*- coding: utf-8 -*-
import os
import time
import random
import torch  # Needed for seeding and checking cuda availability
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import Dataset  # Used by QADataset, QAFinetuneDataset, QADatasetRag
from huggingface_hub import login
import gc  # Garbage Collection
import requests
import openai
import re

# For huggingface Trainer (finetuning)
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

# =====================================================================
# ------------- Hyperparameters, Tokens, and Global Setup -------------
# =====================================================================

# --- DataLoader & Evaluation Settings ---
BATCH_SIZE_GPU = 128
NUM_WORKERS_GPU = os.cpu_count()
BATCH_SIZE_API = 16
NUM_WORKERS_API = 4

# --- LLM Generation Hyperparameters ---
TEMPERATURE = 1e-5
TOP_P = 1
TOP_K = 1
MAX_NEW_TOKENS = 10

# --- Other Settings ---
SEED = 42
INVALID_ANSWER_MARKER = "INVALID"
RESULTS_FILENAME_TEMPLATE = "evaluation_results_{dataset_name}_{model_name}{suffix}.xlsx"

# --- Tokens and API keys (Load from .env file) ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Hugging Face Login ---
def perform_hf_login():
    if not HF_TOKEN:
        print("Warning: Hugging Face token (HF_TOKEN) not found.")
    else:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
        except Exception as e:
            print(f"Error logging into Hugging Face Hub: {e}")

# --- Global Model Info Lookup ---
LLMs_info_df = None
def load_model_info(info_file_path=os.path.join("data", "LLMs_info.xlsx")):
    global LLMs_info_df
    try:
        LLMs_info_df = pd.read_excel(info_file_path)
    except FileNotFoundError:
        print(f"Warning: {info_file_path} not found. Model info unavailable.")
        LLMs_info_df = pd.DataFrame(columns=["model_name", "model_family", "model_size_b_paras"])
    except Exception as e:
        print(f"Error reading {info_file_path}: {e}")
        LLMs_info_df = pd.DataFrame(columns=["model_name", "model_family", "model_size_b_paras"])

def extract_model_info(model_name):
    """Looks up model info. Returns (family, size) or ('Unknown', 0.0)."""
    if LLMs_info_df is None:
        load_model_info()
    try:
        row = LLMs_info_df.loc[LLMs_info_df["model_name"] == model_name]
        if not row.empty:
            return row.iloc[0]["model_family"], row.iloc[0]["model_size_b_paras"]
    except Exception:
        pass
    return "Unknown", 0.0

# --- Randomness Control ---
def set_random_seeds(seed=SEED):
    """Sets random seeds for reproducibility."""
    print(f"Setting random seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"Warning: Could not set CUDA/CuDNN settings: {e}")

# =====================================================================
# --------------------- Original Zero-Shot Code -----------------------
# =====================================================================

def validate_prediction(pred_str):
    """
    Validates the generated prediction.
    Returns the validated string (uppercase) or INVALID if invalid.
    Valid formats: 'Z' or sorted unique letters from 'ABCD'.
    """
    if pred_str is None:
        return INVALID_ANSWER_MARKER
    
    s = pred_str.strip()
    if not s:
        return INVALID_ANSWER_MARKER
    
    if s == 'Z':
        return s
    elif all(c in 'ABCD' for c in s):
        return ''.join(sorted(set(s)))

    s = ' '.join(re.sub('[^A-Za-z]', ' ', str(pred_str)).split()).strip()
    if not s:
        return INVALID_ANSWER_MARKER
    
    s = ' '.join(word.strip() for word in s.split() if not any(ch.islower() for ch in word)).strip()
    if not s:
        return INVALID_ANSWER_MARKER
    
    s = ' '.join(word for word in s.split() if all(ch in 'ABCDZ' for ch in word)).strip()
    if not s:
        return INVALID_ANSWER_MARKER
    
    if s == 'Z':
        return 'Z'
    elif 'Z' in s:
        return INVALID_ANSWER_MARKER
    
    result = ''.join(sorted(set(s))).strip()
    if not result:
        return INVALID_ANSWER_MARKER
    return result

def build_prompt(row, dataset_name=None):
    """
    Standard prompt builder used by your zero-shot pipeline.
    """
    if dataset_name and '775' in dataset_name and 'lowest_200' not in dataset_name:
        prompt = (
            "You are an expert in LEED Green Associate from US Green Council. Answer the question with a single letter or a required number (Choose X) of distinct letters in alphabetical order. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'E', 'F', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif any(name in dataset_name for name in ['Synthetic', 'lowest_200']):
        prompt = (
            "You are an expert in LEED Green Associate from US Green Council. Answer the question with a single letter only. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"           
    elif 'until_id' in dataset_name:
        prompt = (
            "You are an expert in ESG (Environmental, Social, Governance) and Sustainability related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'IPCC' in dataset_name:
        prompt = (
            "You are an expert in IPCC (Intergovernmental Panel on Climate Change) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'GRI' in dataset_name:
        prompt = (
            "You are an expert in GRI (Global Reporting Initiative) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'SASB' in dataset_name:
        prompt = (
            "You are an expert in SASB (Sustainability Accounting Standards Board) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'ISO' in dataset_name:
        prompt = (
            "You are an expert in ISO (International Organization for Standardization) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )   
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'IFRSandISSB' in dataset_name:
        prompt = (
            "You are an expert in IFRS (International Financial Reporting Standards) and ISSB (International Sustainability Standards Board) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'TCFD' in dataset_name:
        prompt = (
            "You are an expert in TCFD (Task Force on Climate-related Financial Disclosures) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'CDP' in dataset_name:
        prompt = (
            "You are an expert in CDP (Carbon Disclosure Project) related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
    elif 'ESGenius' in dataset_name:
        prompt = (
            "You are an expert in ESG (Environmental, Social, Governance) and Sustainability related topics. Answer the question with a single letter based on authoritative knowledge. Each option content is case-sensitive.\n\n"
        )
        prompt += f"Question: {row.get('query', '')}\n"
        prompt += "Options:\n"
        for option in ['A', 'B', 'C', 'D', 'Z']:
            option_text = row.get(option)
            if pd.notna(option_text):
                prompt += f"{option}: {option_text}\n"
                
    prompt += "\nAnswer:"
    return prompt

class QADataset(Dataset):
    """Zero-shot dataset used by your existing pipeline."""
    def __init__(self, dataframe, dataset_name=None):
        self.df = dataframe
        self.dataset_name = dataset_name
        if 'query_id' not in self.df.columns:
            raise ValueError("'query_id' column missing.")
        if 'answer' not in self.df.columns:
            raise ValueError("'answer' column missing.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        query_id = str(row['query_id']).strip()
        prompt = build_prompt(row, dataset_name=self.dataset_name)
        gold_answer = str(row['answer']).strip().upper() if pd.notna(row['answer']) else ""
        return query_id, prompt, gold_answer

# =====================================================================
# --------------------- RAG-Specific Code -----------------------------
# =====================================================================

def retrieve_context_rag(question_str, source_text, top_k=1):
    """
    Minimal RAG retrieval function:
    Splits 'source_text' by newlines, ranks lines by overlap, returns up to 'top_k'.
    """
    if not source_text or not isinstance(source_text, str):
        return ""
    question_words = set(question_str.lower().split())
    lines = [ln.strip() for ln in source_text.split("\n") if ln.strip()]
    scored_lines = []
    for line in lines:
        line_words = set(line.lower().split())
        overlap = len(question_words.intersection(line_words))
        scored_lines.append((overlap, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    best_lines = [item[1] for item in scored_lines[:top_k] if item[0] > 0]
    return "\n".join(best_lines) if best_lines else ""

class QADatasetRag(Dataset):
    """
    RAG dataset that fetches 'source_text'.
    """
    def __init__(self, dataframe, dataset_name=None):
        self.df = dataframe
        self.dataset_name = dataset_name
        if 'query_id' not in self.df.columns:
            raise ValueError("'query_id' column missing.")
        if 'answer' not in self.df.columns:
            raise ValueError("'answer' column missing.")
        if 'source_text' not in self.df.columns:
            print("WARNING: 'source_text' column missing for RAG. Setting to empty string.")
            self.df['source_text'] = ""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        query_id = str(row['query_id']).strip()
        gold_answer = str(row['answer']).strip().upper() if pd.notna(row['answer']) else ""
        prompt = build_prompt(row, dataset_name=self.dataset_name)
        row_source_text = str(row.get('source_text', ""))

        return query_id, prompt, gold_answer, row_source_text

# =====================================================================
# ------------------- Minimal Finetuning Additions --------------------
# =====================================================================

class QAFinetuneDataset(Dataset):
    """
    Minimal supervised fine-tuning dataset for a Causal LM.
    We'll treat each row's (prompt, answer) as input->label.
    Optionally, incorporate source_text if you want it in the prompt.
    """
    def __init__(self, dataframe, dataset_name=None, use_source_text=True):
        self.df = dataframe
        self.dataset_name = dataset_name
        self.use_source_text = use_source_text
        if 'query' not in self.df.columns:
            raise ValueError("'query' column missing for question.")
        if 'answer' not in self.df.columns:
            raise ValueError("'answer' column missing for label.")
        if self.use_source_text and 'source_text' not in self.df.columns:
            print("WARNING: 'source_text' column missing but use_source_text=True. Will treat as empty.")
            self.df['source_text'] = ""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        question_str = str(row['query']).strip()
        gold_str = str(row['answer']).strip()
        source_str = ""
        if self.use_source_text:
            source_str = str(row.get('source_text', "")).strip()

        # For minimal approach:
        # E.g. "Context:\n{source_str}\n\nQuestion: {question_str}\nAnswer: {gold_str}"
        input_text = f"Context:\n{source_str}\n\nQuestion: {question_str}\nAnswer:"
        label_text = gold_str

        return input_text, label_text

def finetune_collate_fn(batch, tokenizer, max_length=512):
    """
    A simple collate function for finetuning:
    1) Concatenate input_text + label_text
    2) Tokenize
    3) Use the same tokens as labels for causal LM training
    """
    input_texts = []
    for input_text, label_text in batch:
        combined = input_text + " " + label_text  # e.g. "Context: ... Answer: correct_label"
        input_texts.append(combined)

    tokenized = tokenizer(
        input_texts,
        truncation=True,
        max_length=max_length,
        padding=True
    )
    # For causal LM, labels are the same as input_ids (shifted inside the model).
    tokenized["labels"] = tokenized["input_ids"].copy()
    return {k: torch.tensor(v) for k, v in tokenized.items()}

def train_finetune_model(
    df,
    dataset_name,
    base_model_name,
    output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2
):
    """
    Minimal function to finetune a base causal model on your CSV Q-A data.
    Saves the resulting model in `output_dir`.
    """
    print(f"Loading base model for finetune: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # If no pad token, set it
    if tokenizer and not tokenizer.pad_token:
        if tokenizer.eos_token:
            print("Setting pad_token to eos_token for finetuning.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Adding new [PAD] token for finetuning.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    # Prepare the dataset
    finetune_dataset = QAFinetuneDataset(df, dataset_name, use_source_text=True)

    # Optionally split train/val
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(finetune_dataset.df, test_size=0.1, random_state=SEED)
    ds_train = QAFinetuneDataset(train_df, dataset_name, use_source_text=True)
    ds_eval = QAFinetuneDataset(eval_df, dataset_name, use_source_text=True)

    def data_collator(batch):
        return finetune_collate_fn(batch, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="epoch",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator
    )

    print("Starting training on Q-A pairs...")
    trainer.train()

    print("Evaluating on hold-out set...")
    eval_metrics = trainer.evaluate()
    print(f"Eval metrics: {eval_metrics}")

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Finetuning complete. Model artifacts saved.")


# =====================================================================
# ---------------------- DAPT Additions -------------------------------
# =====================================================================
class SourceTextDataset(Dataset):
    """
    For domain-adaptive pre-training:
    We simply read each row's `source_text` and train the model to predict 
    the next token (i.e., standard LM objective).
    No Q-A pairs needed.
    """
    def __init__(self, df, text_column="source_text"):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' missing in dataset for DAPT.")
        # Convert to string and fill NaN
        self.texts = df[text_column].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def _dapt_collate_fn(texts, tokenizer, max_length=512):
    """
    Collator for domain-adaptive pre-training:
    1) Tokenize each `source_text`
    2) Use the same tokens for labels (causal LM objective)
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return {k: torch.tensor(v) for k, v in encodings.items()}

def train_dapt_model(
    df,
    base_model_name,
    output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4
):
    """
    Domain-Adaptive Pre-Training:
    Fine-tune a base model on the `source_text` column alone,
    to let the model absorb domain knowledge. 
    No question-answer usage here.
    """
    print(f"Loading base model for DAPT: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Build the dataset
    dapt_dataset = SourceTextDataset(df, text_column="source_text")

    # Define trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        # Some basic logging / saving config
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        do_train=True,
        do_eval=False  # pure LM training, no eval unless you have a val set
    )

    def dapt_data_collator(batch):
        return _dapt_collate_fn(batch, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dapt_dataset,
        data_collator=dapt_data_collator
    )

    print("Starting domain-adaptive pre-training (DAPT) ...")
    trainer.train()
    print("DAPT training complete.")

    print(f"Saving DAPT model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("DAPT artifacts saved successfully.")


# =====================================================================
# ------------------- Common Utility Functions ------------------------
# =====================================================================

def compute_summary_metrics(eval_df, model_name):
    """Compute transposed DataFrame with summary metrics."""
    model_ans_col = f"{model_name}_ans"
    non_empty = eval_df[eval_df['answer'].notna()]
    total = len(non_empty)
    num_correct = 0
    num_invalid = 0
    for _, row in non_empty.iterrows():
        gold = str(row.get("answer", "")).strip().upper()
        model_ans = str(row.get(model_ans_col, "")).strip().upper()
        if gold:
            if model_ans == INVALID_ANSWER_MARKER or model_ans == "":
                num_invalid += 1
            elif model_ans == gold:
                num_correct += 1
    num_wrong = total - (num_correct + num_invalid)
    model_acc = num_correct / total if total > 0 else 0.0
    ratio_invalid = num_invalid / total if total > 0 else 0.0
    ratio_wrong = num_wrong / total if total > 0 else 0.0

    metrics_dict = {
         "total_Q": f"{total}",
         "num_correct_Q": f"{num_correct}",
         "num_invalid_Q": f"{num_invalid}",
         "num_wrong_Q": f"{num_wrong}",
         "model_acc": f"{model_acc:.4f}",
         "ratio_invalid_Q": f"{ratio_invalid:.4f}",
         "ratio_wrong_Q": f"{ratio_wrong:.4f}",
         "model_acc_pct": f"{model_acc * 100:.2f}%",
         "ratio_invalid_Q_pct": f"{ratio_invalid * 100:.2f}%",
         "ratio_wrong_Q_pct": f"{ratio_wrong * 100:.2f}%"
    }

    summary_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    return summary_df

def save_eval_df_to_excel(eval_df, excel_file, model_name):
    """
    Saves eval_df to Excel with a summary sheet and a details sheet.
    """
    try:
        file_dir = os.path.dirname(excel_file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        parent_dir = os.path.dirname(excel_file) if os.path.dirname(excel_file) else '.'
        if not os.access(parent_dir, os.W_OK):
            print(f"Error: No write permission for directory '{parent_dir}'. Skipping save.")
            return

        summary_df = compute_summary_metrics(eval_df, model_name)

        raw_col = f"{model_name}_raw_res"
        ans_col = f"{model_name}_ans"
        if raw_col not in eval_df.columns:
            eval_df[raw_col] = pd.NA
        if ans_col not in eval_df.columns:
            eval_df[ans_col] = pd.NA

        remaining_cols = [col for col in eval_df.columns if col not in [raw_col, ans_col]]
        details_order = [raw_col, ans_col] + remaining_cols
        details_df = eval_df[details_order]

        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, index=False, sheet_name='summary')
            details_df.to_excel(writer, index=False, sheet_name='details')

            workbook  = writer.book
            wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

            summary_ws = writer.sheets['summary']
            summary_ws.set_column(0, summary_df.shape[1]-1, 20, wrap_format)
            summary_ws.freeze_panes(1, 0)

            details_ws = writer.sheets['details']
            details_ws.set_column(0, details_df.shape[1]-1, 20, wrap_format)
            details_ws.freeze_panes(1, 0)
    except PermissionError:
        print(f"\nError: Permission denied when trying to save {excel_file}.")
        print("Please ensure the file is not open and you have write permissions.")
    except Exception as e:
        print(f"\nError saving Excel file {excel_file}: {type(e).__name__} - {e}")

def record_result(eval_df, qid, model_name, raw_response, prediction):
    """
    Updates eval_df in memory.
    """
    raw_col = f"{model_name}_raw_res"
    ans_col = f"{model_name}_ans"
    if raw_col not in eval_df.columns:
        eval_df[raw_col] = pd.NA
    if ans_col not in eval_df.columns:
        eval_df[ans_col] = pd.NA
    str_qid = str(qid).strip()
    if "query_id" in eval_df.columns:
        mask = eval_df["query_id"].astype(str) == str_qid
        match_indices = eval_df.index[mask].tolist()
        if len(match_indices) > 0:
            idx = match_indices[0]
            if len(match_indices) > 1:
                print(f"Warning: Multiple rows found for query_id '{str_qid}'. Updating first instance at index {idx}.")
            eval_df.loc[idx, raw_col] = raw_response
            eval_df.loc[idx, ans_col] = prediction
    else:
        print(f"CRITICAL ERROR: 'query_id' column not found in record_result for qid {str_qid}.")
    return eval_df

def load_dataset(dataset_folder="data", filename=None):
    """Loads the QA dataset."""
    if filename is None:
        print("Error: No filename provided. Please provide a CSV filename.")
        return None, None
    dataset_path = os.path.join(dataset_folder, filename)
    print(f"Attempting to load dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded. Shape: {df.shape}")
        if "query_id" not in df.columns:
            raise ValueError("'query_id' missing.")
        if "answer" not in df.columns:
            raise ValueError("'answer' missing.")
        df["query_id"] = df["query_id"].astype(str).str.strip()
        if df["query_id"].duplicated().any():
            print("Warning: Duplicate 'query_id' values found in dataset. Results for duplicates may be overwritten.")
        return df, os.path.splitext(filename)[0]
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {dataset_path}. Exiting.")
        return None, None
    except ValueError as ve:
        print(f"Error: {ve}. Exiting.")
        return None, None
    except Exception as e:
        print(f"Error reading dataset CSV {dataset_path}: {e}. Exiting.")
        return None, None

def load_or_initialize_eval_df(base_df, dataset_name, model_name, results_folder="results", results_filename=None, rag=False, finetune=False):
    """
    Loads existing results or initializes a new DataFrame.
    """
    model_name_safe = model_name.replace("/", "_")
    if results_filename is None:
        suffix = ""
        if rag:
            suffix = "_rag"
        elif finetune:
            suffix = "_finetune"
        results_filename = RESULTS_FILENAME_TEMPLATE.format(
            dataset_name=dataset_name,
            model_name=model_name_safe,
            suffix=suffix
        )
    evaluation_excel_file = os.path.join(results_folder, results_filename)
    print(f"Using results file: {evaluation_excel_file}")
    os.makedirs(results_folder, exist_ok=True)

    if os.path.exists(evaluation_excel_file):
        print("Existing results file found. Loading details sheet...")
        try:
            eval_df = pd.read_excel(evaluation_excel_file, sheet_name='details')
            print(f"Loaded existing details. Shape: {eval_df.shape}")
            if "query_id" not in eval_df.columns:
                print("Error: 'query_id' column missing in existing file! Re-initializing.")
                eval_df = base_df.copy()
                eval_df["query_id"] = eval_df["query_id"].astype(str).str.strip()
                save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)
            else:
                eval_df["query_id"] = eval_df["query_id"].astype(str).str.strip()
        except Exception as e:
            print(f"Error loading existing Excel file '{evaluation_excel_file}': {e}")
            print("Initializing fresh DataFrame.")
            eval_df = base_df.copy()
            eval_df["query_id"] = eval_df["query_id"].astype(str).str.strip()
            save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)
    else:
        print("No existing results file found. Initializing fresh DataFrame.")
        eval_df = base_df.copy()
        eval_df["query_id"] = eval_df["query_id"].astype(str).str.strip()
        print("Saving initial DataFrame structure...")
        save_eval_df_to_excel(eval_df, evaluation_excel_file, model_name)

    if eval_df.empty or "query_id" not in eval_df.columns:
        print("Error: Eval DataFrame is invalid after load/initialization.")
        return None, None

    return eval_df, evaluation_excel_file

def check_if_skip_model(model_name, dataset_name, results_folder="results", rag=False, finetune=False):
    """
    Checks if an evaluation file for a model already exists.
    """
    model_name_safe = model_name.replace("/", "_")
    suffix = ""
    if rag:
        suffix = "_rag"
    elif finetune:
        suffix = "_finetune"
    results_filename = RESULTS_FILENAME_TEMPLATE.format(
        dataset_name=dataset_name,
        model_name=model_name_safe,
        suffix=suffix
    )
    evaluation_excel_file = os.path.join(results_folder, results_filename)
    if os.path.exists(evaluation_excel_file):
        print(f"Results file already exists for '{model_name}' at {evaluation_excel_file}. Skipping evaluation.")
        return True
    return False

def report_model_accuracy(model_name, eval_df):
    """Print final accuracy from the summary metrics."""
    summary_df = compute_summary_metrics(eval_df, model_name)
    print(f"\n--- Final Summary for {model_name} ---")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}: {row['Value']}")

def cleanup_gpu_resources(generator=None, model=None, tokenizer=None):
    """Deletes objects and tries to clear CUDA cache."""
    del generator
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error clearing CUDA cache: {e}")
