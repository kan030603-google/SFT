from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import sys
import os
import io

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "D:\\ti_cache" 
os.environ["TRITON_CACHE_DIR"] = "D:\\triton_cache"
# 解决 Windows 下打印 emoji 报错的问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ============== 全局配置和函数定义（保持不变，放在最外层） ====================
local_model_path = 'D:/DATA/MODEL/pretrained/unsloth/Qwen3-8B-unsloth-bnb-4bit'
dataset_path = "./data/keywords_data_train.jsonl"

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
            ])
    return {"conversations": conversations}

def format_func(example):
    formatted_texts = []
    # 注意：这里的 tokenizer 需要在 main 中传递或者作为全局变量小心使用
    # 在 Windows spawn 模式下，建议在 main 内部处理，或者确保 tokenizer 已定义
    # 为简单起见，这里假设 tokenizer 实际上是在 main 中定义的，
    # 但由于 spawn 机制，format_func 在子进程运行时可能找不到 tokenizer。
    # **稳妥的做法是把 format_func 也移到 main 里面，或者使用 partial 传递 tokenizer**
    pass 
    # (下文修复代码中，我会保留原本逻辑，但要在 main 里运行)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 核心修改：Windows 必须加这行，否则 100% 报错
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if __name__ == "__main__":
    
    # ============== 1、加载模型、tokenizer ====================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=2048,
        device_map="cuda:0",
        dtype=None,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print(model)

    # ===================== 2.数据加载与格式转换 ==========================
    # 重新定义 format_func 以确保它可以访问 tokenizer (闭包)
    def format_func(example):
        formatted_texts = []
        for conv in example['conversations']:
            formatted_texts.append(
                tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text":formatted_texts}

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 关键点：在 Windows 上，num_proc 如果大于 1 也会触发多进程问题
    # 如果加上了 if __name__ == "__main__"，这里可以用多核
    dataset = dataset.map(
        convert_to_qwen_format,
        batched=True,
        remove_columns=dataset.column_names
    )

    formatted_dataset = dataset.map(
        format_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    # ==================== 3.使用trl库的训练器 ====================
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,

            
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            report_to = "none",
            # Windows 下 DataLoader 的 worker 数量建议设为 0 或 1，防止报错
            dataloader_num_workers = 0, 
        ),
    )

    # 显示当前内存统计信息
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # 开始训练
    trainer_stats = trainer.train()

    # 显示最终内存和时间统计信息
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training = {lora_percentage} %.")

    # ==================== 4.保存训练结果 ====================================
    model.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-lora-adapter-unsloth")
    tokenizer.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-lora-adapter-unsloth")