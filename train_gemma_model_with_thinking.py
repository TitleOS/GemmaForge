# -*- coding: utf-8 -*-
"""
Prerequisites:
- PyTorch
- Unsloth
- Transformers (`pip install transformersevaluate rouge_score bert_score sentencepiece==4.56.2`)
- TRL (`pip install --no-deps trl==0.22.2`)
- Datasets (`pip install "datasets>=3.4.1,<4.0.0"`)
- Accelerate, bitsandbytes, peft

Example Usage (including GGUF quantization and upload):
python train_gemma.py \
    --max_seq_length 2048 \
    --dataset "acon96/Home-Assistant-Requests" \
    --r 16 \
    --quantize_gguf \
    --repo_name "YourHuggingFaceUsername/HomeGem" \
    --token "hf_YOUR_TOKEN_HERE"

    python train.py --max_seq_length 8192 --model_name "unsloth/gemma-3-27b-it-unsloth-bnb-4bit" --dataset "Jackrong/Natural-Reasoning-gpt-oss-120B-S1" --token "hf_mytoken" --r 32 --quantize_gguf --repo_name "TitleOS/MindStone27B" --steps 340
"""
import unsloth
import argparse
import torch
import random
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TextStreamer

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats


def main():
    """
    Main function to run the Gemma 3 fine-tuning process.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a Gemma 3 model using Unsloth."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for the model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset name to use for training."
    )
    parser.add_argument(
        "--r",
        type=int,
        default=16,
        help="The rank for the PEFT model's LoRA configuration."
    )
    parser.add_argument(
        "--quantize_gguf",
        action="store_true",
        help="Quantize the model to GGUF Q8_0 format and upload if repo_name and token are provided."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Your Hugging Face token for uploading the model."
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="The Hugging Face repository name to push the model to (e.g., 'user/repo')."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default='0245',
        help="The seed used for various RNGs, this allows replication across sessions."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default='120',
        help="The number of iterations or steps to finetune the model for. Default = 120."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        help="The model to finetune. Use a 4bit model such as Unsloth's if passing --4bit"
    ),
    parser.add_argument(
        "--four_bit",
        type=bool,
        default="True",
        help="Enables finetuning the model as 4 bit quantization."
    )

    args = parser.parse_args()

    # --- 1. Load Model and Tokenizer ---
    print("Step 1: Loading the base model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
        #model_name="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=True
    )
    print("Model and tokenizer loaded successfully.")

    # --- 2. Configure PEFT (LoRA) ---
    print("\nStep 2: Configuring the model with PEFT LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.r,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    print(f"PEFT configured with r={args.r} and lora_alpha={args.r}.")

    # --- 3. Data Preparation ---
    print("\nStep 3: Preparing the dataset...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    def formatting_prompts_func(examples):
        questions = examples["question"]
        cot_reasoning = examples["CoT_reasoning"]
        answers = examples["anwser"]
    
        texts = []
        for question, reasoning, answer in zip(questions, cot_reasoning, answers):
            #Generate a way to incorporate the /no_thinking command. By cutting our data set in half, and training only those who are in group B vwithout the reasoning tags, the model should develop a concept to have controllable thinking.
            no_thinking_sample = random.choice([True, False])
            if no_thinking_sample == True:
                model_response = f"{answer}"
                messages = [
                {"role": "user", "content": "/no_thinking " + question},
                {"role": "model", "content": model_response},
                ]
            elif no_thinking_sample == False:
                model_response = f"<reasoning>\n{reasoning}\n</reasoning>\n{answer}"
                # Create the structured conversation for the template
                messages = [
                    {"role": "user", "content": question},
                    {"role": "model", "content": model_response},
                ]   

            # Apply the chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset(args.dataset, split="train")
    dataset = standardize_data_formats(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("Dataset prepared and formatted.")
    print(f"Sample formatted data:\n{dataset[100]['text']}")

    # --- 4. Configure and Run Training ---
    print("\nStep 4: Setting up and starting the training process...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            dataset_text_field="text",
            warmup_steps=5,
            max_steps=args.steps,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir="outputs"
        ),
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU: {gpu_stats.name}. Max memory: {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")

    trainer_stats = trainer.train()

    # --- 5. Post-Training Stats & Evaluation Loss Monitoring ---
    print("\nStep 5: Displaying post-training statistics and evaluation loss...")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    
    print(f"{trainer_stats.metrics['train_runtime']:.2f} seconds used for training.")
    print(f"Peak reserved memory: {used_memory} GB.")
    print(f"Peak memory for training: {used_memory_for_lora} GB.")

    # --- 7. Save and Upload Model ---
    if args.repo_name and args.token:
        print(f"\nStep 7: Uploading model to Hugging Face Hub repo: {args.repo_name}")

        try:
            print("Pushing LoRA adapters...")
            model.push_to_hub(args.repo_name, token=args.token)
            tokenizer.push_to_hub(args.repo_name, token=args.token)
            print("Successfully pushed LoRA adapters.")
        except Exception as e:
            print(f"Error pushing LoRA adapters: {e}")

        try:
            print("Pushing merged 16-bit model...")
            model.push_to_hub_merged(args.repo_name, tokenizer, save_method="merged_16bit", token=args.token)
            print("Successfully pushed merged 16-bit model.")
        except Exception as e:
            print(f"Error pushing merged 16-bit model: {e}")

        if args.quantize_gguf:
            print("Quantizing and pushing GGUF Q8_0 model...")
            try:
                model.push_to_hub_gguf(args.repo_name, tokenizer, quantization_method="q8_0", token=args.token)
                print("Successfully pushed GGUF model.")
            except Exception as e:
                print(f"Failed to create and upload GGUF model. This can happen due to memory constraints. Error: {e}")

    else:
        print("\nStep 7: Saving models locally (no token/repo_name provided).")
        print("Saving LoRA adapters to './lora_model'...")
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
        print("LoRA model saved.")

        if args.quantize_gguf:
            print("Quantizing and saving GGUF Q8_0 model locally...")
            try:
                model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q8_0")
                print("GGUF model saved to './gguf_model'.")
            except Exception as e:
                print(f"Failed to create local GGUF model. This can happen due to memory constraints. Error: {e}")
        
        print("\nTo upload to Hugging Face Hub, provide --repo_name and --token arguments.")

    print("\nScript finished successfully!")


if __name__ == "__main__":
    main()