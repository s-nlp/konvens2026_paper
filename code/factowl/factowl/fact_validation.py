#!/usr/bin/env python3
"""
Atomic Fact Validator using VLLM
Validates whether atomic facts are specific to a single entity or generic.

Usage:
    python validate_facts.py input.csv output.csv --claim-column "claim" --model "meta-llama/Llama-3.3-70B-Instruct" --num-examples 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_few_shot_examples():
    """Return all available few-shot examples."""
    examples = [
        # Example 1: Generic statement about storms
        {
            "user": "Atomic fact: The storm highlighted the need for more robust infrastructure and emergency response systems.",
            "assistant": "This is a generic statement that could apply to many hurricanes or natural disasters, not unique to Hurricane Ian; BAD"
        },
        # Example 2: Specific hurricane with locations
        {
            "user": "Atomic fact: The impact of Hurricane Ian was felt far beyond the immediate storm surge zone, with heavy rainfall and flooding reported in central and eastern Florida.",
            "assistant": "This fact specifies unique impacts and locations directly tied to Hurricane Ian; GOOD"
        },
        # Example 3: Pronoun without clear reference
        {
            "user": "Atomic fact: He graduated from Amherst College and the University of Arizona.",
            "assistant": "This claim contains a pronoun and can refers to a variety of entities; BAD"
        },
        # Example 4: Specific statistics with date
        {
            "user": "Atomic fact: In 2019 the US Police were 20.8 times more likely to kill than be killed by a criminal.",
            "assistant": "This fact specifies unique statistics and date directly tied to US Police; GOOD"
        },
        # Example 5: Named person with specific achievement
        {
            "user": "Atomic fact: Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields.",
            "assistant": "This fact explicitly mentions Marie Curie and describes unique achievements specific to her; GOOD"
        },
        # Example 6: Generic description without entity
        {
            "user": "Atomic fact: The company reported record profits in the third quarter.",
            "assistant": "This statement lacks entity specification and could apply to any company in any year; BAD"
        },
        # Example 7: Specific event with date and location
        {
            "user": "Atomic fact: The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, at the Sea of Tranquility.",
            "assistant": "This fact provides specific mission name, date, and location unique to Apollo 11; GOOD"
        },
        # Example 8: Vague temporal reference
        {
            "user": "Atomic fact: The election resulted in a narrow victory for the incumbent party.",
            "assistant": "This statement is too generic without specifying which election, country, or year; BAD"
        },
        # Example 9: Specific legislation with details
        {
            "user": "Atomic fact: The Affordable Care Act, signed by President Obama in 2010, expanded healthcare coverage to millions of Americans.",
            "assistant": "This fact names the specific act, the president who signed it, and the year, making it uniquely identifiable; GOOD"
        },
        # Example 10: Generic scientific statement
        {
            "user": "Atomic fact: The research showed promising results in early clinical trials.",
            "assistant": "This statement could apply to any research study and lacks specific identification; BAD"
        }
    ]
    return examples


def create_prompt(claim, tokenizer, num_examples=4):
    """Create a prompt for fact validation with configurable number of examples."""

    # System message
    messages = [
        {
            "role": "system",
            "content": """You are an expert in verifying atomic facts extracted from Wikipedia pages for specificity to a single entity. Your task is to determine if a given atomic fact refers exclusively to the specified topic/entity and has no multiple meanings or applicability to other similar entities without direct reference to the topic.\
Rules:\
- The fact must be unique or directly tied to the topic. It should not be a generic statement that could apply to many entities (e.g., locations, people, events). Entities must be explicitly stated. The fact may not be true, correct, or may be debatable, but that is not important; it must only comply with rules. \
- If the fact mentions the topic explicitly or describes details unique to it, it's GOOD.\
- If the fact is vague, general, or could be said about many similar entities, it's BAD.\
- Output exactly in this format: a short explanation (1-2 sentences) followed by a semicolon and a choice of "GOOD" or "BAD"."""
        }
    ]

    # Add few-shot examples based on num_examples parameter
    if num_examples > 0:
        all_examples = get_few_shot_examples()
        selected_examples = all_examples[:min(num_examples, len(all_examples))]

        for example in selected_examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

    # Add the actual claim to validate
    messages.append({"role": "user", "content": f"Atomic fact: {claim}"})

    # Check if model supports thinking mode
    # enable_thinking = "Llama-3" in model_name or "llama-3" in model_name
    enable_thinking = True

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt


def process_outputs(outputs):
    """Process model outputs to extract classifications."""
    processed = []
    clean_labels = []
    is_good = []

    for output in outputs:
        text = output.outputs[0].text.strip()

        # Remove thinking tags if present
        if 'think>' in text:
            text = text.split('think>')[-1].strip()

        processed.append(text)

        # Extract label (GOOD/BAD)
        label = text.split(";")[-1].strip() if ";" in text else text
        clean_labels.append(label)

        # Boolean classification
        is_good.append('GOOD' in label)

    return processed, clean_labels, is_good


def validate_facts(input_tsv, long_output_file, clean_output_file, vllm_model, tokenizer, vllm_sampling_params,
                   claim_column='atom', num_examples=10):
    df = pd.read_csv(input_tsv, sep='\t')

    # Create prompts
    print("Creating prompts...")
    prompts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        claim = row[claim_column]
        if pd.notna(claim):  # Skip NaN values
            prompt = create_prompt(claim, tokenizer, num_examples)
            prompts.append(prompt)
        else:
            prompts.append("")  # Placeholder for NaN

    df['primed_text'] = prompts

    # Filter out empty prompts for generation
    valid_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_indices]

    # Generate responses
    print(f"Generating responses for {len(valid_prompts)} valid claims...")
    outputs = vllm_model.generate(valid_prompts, vllm_sampling_params)

    # Process outputs
    print("Processing outputs...")
    processed, clean_labels, is_good = process_outputs(outputs)

    # Map results back to dataframe (accounting for NaN values)
    df['out'] = ""
    df['out_clean'] = ""
    df['is_good'] = None

    for idx, orig_idx in enumerate(valid_indices):
        df.loc[orig_idx, 'out'] = processed[idx]
        df.loc[orig_idx, 'out_clean'] = clean_labels[idx]
        df.loc[orig_idx, 'is_good'] = is_good[idx]

    # Save results
    print(f"Saving long results to {long_output_file}")
    df.to_csv(long_output_file, sep='\t', index=False)
    clean_df = df[df["is_good"]]
    clean_df["sample_id"] = list(range(clean_df.shape[0]))
    clean_df[["sample_id","topic","atom","is_supported","label"]].to_csv(clean_output_file, sep='\t', index=False)

    # Print summary
    total_processed = len(valid_indices)
    total_good = sum(is_good)
    total_bad = total_processed - total_good

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Processed: {total_processed}")
    print(f"Skipped (NaN): {len(df) - total_processed}")
    print(f"GOOD: {total_good} ({total_good / total_processed * 100:.1f}%)")
    print(f"BAD: {total_bad} ({total_bad / total_processed * 100:.1f}%)")
    print(f"Long output saved to: {long_output_file}")
    print(f"Clean output saved to: {clean_output_file}")

    return df

