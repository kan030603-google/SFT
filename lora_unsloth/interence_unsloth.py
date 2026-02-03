from torch.return_types import return_types
from unsloth import FastLanguageModel
from transformers import TextStreamer

local_model_path = "D:/DATA/MODEL/qwen/unsloth/Qwen3-8B-unsloth-bnb-4bit"
lora_adapter_path = "D:/DATA/MODEL/LoRA2qwen/outputs/Qwen3-8B-sft-lora-adapter-unsloth"

# 1、加载基座模型、tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=2048,  # 支持32K+长上下文
    device_map="auto",
    dtype=None,  # 自动选择最优精度
    load_in_4bit=True,  # 4bit量化节省70%显存
)

# 2、注入lora适配器
# model.load_adapter(lora_adapter_path)
# 启用unsloth推理加速
FastLanguageModel.for_inference(model)
model.eval()

# 3、构造数据  {text:<im_start>\nsystem....<im_end>} ==> tokenizer.apply_chat_template
messages = [
    {"role": "user", "content": "关键词识别：\n梯度功能材料是基于一种全新的材料设计概念而开发的新型功能材料.陶瓷-金属FGM的主要结构特点是各梯度层由不同体积浓度的陶瓷和金属组成,材料在升温和降温过程中宏观梯度层间产生热应力,每一梯度层中细观增强相和基体的热物性失配将产生单层热应力,从而导致材料整体的破坏.采用云纹干涉法,对具有四个梯度层的SiC/A1梯度功能材料分别在机载、热载及两者共同作用下进行了应变测试,分别得到了这三种情况下每梯度层同一位置的纵向应变,横向应变和剪应变值."}
]
format_messages = tokenizer.apply_chat_template(
                messages,
                tokenize=False,                  # 训练时部分词，true返回的是张量
                add_generation_prompt=True,    # 训练期间要关闭，如果是推理则设为True
            )

# 4、调用tokenizer得到input
inputs = tokenizer(format_messages,return_tensors='pt').to(model.device)


# 5、调用model.generate()
outputs = model.generate(**inputs,max_new_tokens=1024)

response=tokenizer.decode(outputs[0],skip_special_tokens=True)
# 只输出回答部分

print(response)