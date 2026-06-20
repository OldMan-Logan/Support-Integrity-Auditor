import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

def build_prompt(subject, description):

    return f"""
You are an expert customer support auditor.

Your task is ONLY to estimate the intrinsic severity of the support ticket.

Ignore any assigned priority.

Classify into exactly one:

Low
Medium
High
Critical

Return ONLY JSON.

{{
    "severity": "...",
    "confidence": 0.0,
    "reason": "..."
}}

Ticket Subject:
{subject}

Ticket Description:
{description}
"""

import json

def get_llm_severity(subject, description):

    prompt = build_prompt(subject, description)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    parts = response.splitlines()
    del parts[0]
    del parts[-1]
    new_response = "\n".join(parts)

    try:
        result = json.loads(new_response)
    except:
        result = {
            "severity": "Medium",
            "confidence": 0.5,
            "reason": new_response
        }

    return result
