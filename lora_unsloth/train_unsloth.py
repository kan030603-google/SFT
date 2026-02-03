from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

"""
nohup python -u train_unsloth.py >20250818_1.log 2>&1 &
"""


# ============== 1、加载模型、tokenizer ====================================
local_model_path = 'D:/DATA/MODEL/pretrained/unsloth/Qwen3-8B-unsloth-bnb-4bit'
dataset_path = "./data/keywords_data_train.jsonl"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=2048,  # 支持32K+长上下文
    device_map="cuda:0",
    dtype=None,  # 自动选择最优精度
    load_in_4bit=True,  # 4bit量化节省70%显存
    load_in_8bit=False,
    full_finetuning=False
)


model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=32,  # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # rank stabilized LoRA
    loftq_config=None,  # LoftQ

)


print(model)

# # ===================== 2.数据加载与格式转换 ==========================
def convert_to_qwen_format(example):
    """
    {"conversation_id": 612, "category": "", "conversation": [{"human": "", "assistant": ""}], "dataset": ""}
    :return:
    """
    conversations = []
    for conv_list in example['conversation']:
        for conv in conv_list:
            conversations.append([
                {"role": "user", "content": conv['human'].strip()},
                {"role": "assistant", "content": conv['assistant'].strip()},

            ]
            )
    return {"conversations": conversations}


def format_func(example):
    formatted_texts = []
    for conv in example['conversations']:
        formatted_texts.append(
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,         # 训练时部分词，true返回的是张量
                add_generation_prompt=False,    # 训练期间要关闭，如果是推理则设为True
            )
        )

    return {"text":formatted_texts}


dataset = load_dataset("json", data_files=dataset_path,split="train")
# dataset = dataset.shuffle(seed=43).select(range(1000))
dataset = dataset.map(
    convert_to_qwen_format,
    batched=True,
    remove_columns=dataset.column_names
)
# print(dataset[0])

formatted_dataset = dataset.map(
    format_func,
    batched=True,
    remove_columns=dataset.column_names
)
# print(formatted_dataset[0])



# ==================== 3.使用trl库的训练器 ====================
trainer = SFTTrainer(
    model = model,
    # processing_class = tokenizer,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

# 显示当前内存统计信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats=trainer.train()

# 显示最终内存和时间统计信息
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ==================== 4.保存训练结果 ====================================
# 只保存lora适配器参数
model.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-lora-adapter-unsloth")
tokenizer.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-lora-adapter-unsloth")

# model.save_pretrained_merged("/root/autodl-tmp/outputs/Qwen3-8B-sft-fp16", tokenizer, save_method = "merged_16bit",)
# model.save_pretrained_merged("/root/autodl-tmp/outputs/Qwen3-8B-sft-int4", tokenizer, save_method = "merged_4bit",)