#step1  模型加载
from transformers import AutoTokenizer, AutoModelForCausalLM
#model_name = "G:/finetune/qwen1.5b"
model_name = "G:/finetune/qwen7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("--模型加载完成--")

#step2  数据集制作
import json
from data_loader import samples
with open("dataset.jsonl","w",encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    #else:
        #print("--数据集制作完成--")

#step3 准备训练集和测试集
from datasets import load_dataset
dataset = load_dataset("json",data_files={"train":"dataset.jsonl"},split="train")
#设置15%测试
train_test_split = dataset.train_test_split(test_size=0.15)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
#print(f"训练集大小: {len(train_dataset)}")
#print(f"测试集大小: {len(eval_dataset)}")
print("--训练集测试集准备完成")

#step4  tokenizer处理工具
def tokenizer_function(data):
    texts = [f"判断以下是否是刷单文本，并给出判断依据：\n{prompt}\n{completion}" for prompt,completion in zip(data["prompt"],data["completion"])]
    tokens = tokenizer(texts,padding="max_length",truncation=True,max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function,batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function,batched=True)

print("--完成tokenizing--")
#print(tokenized_train_dataset[0])

#step5  量化设置
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,device_map="auto")
print("--量化模型加载成功--")

#step6  lora微调设置
from peft import LoraConfig,get_peft_model,TaskType
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
print("--lora微调设置完成--")

#step7  设置训练参数
from transformers import TrainingArguments,Trainer
training_args = TrainingArguments(
    #output_dir="./finetuned_models",
    output_dir="./finetuned_models_7b",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=2e-4,
    logging_dir="./logs",
    #run_name="qwen1.5b_finetune"
    run_name="qwen7b_finetune"
)
print("--训练参数设置完成--")

#step8  定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)
print("--开始训练--")
trainer.train()
print("--训练完成--")

#step9  保存模型

#lora模型的保存
#save_path = "./saved_lora_model"
save_path = "./saved_lora_model_7b"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("--lora模型保存完成--")

#全量模型的保存
#final_save_path = "./saved_full_model"
final_save_path = "./saved_full_model_7b"
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
model = PeftModel.from_pretrained(base_model,save_path)
model = model.merge_and_unload()
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("--全量模型保存完成--")






























'''import torch
from fraud_detection_model import EcommerceFraudDetector
from data_loader import create_data_loaders
from train_and_evaluate import train_model, evaluate_model, predict
import argparse


def main():
    parser = argparse.ArgumentParser(description='电商刷单异常行为判定器')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--test_file', type=str, required=True, help='测试数据文件路径')
    parser.add_argument('--output_file', type=str, default='predictions.json', help='预测结果输出文件')
    parser.add_argument('--model_save_path', type=str, default='fraud_detection_model.pt', help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU训练')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'使用设备: {device}')

    # 创建数据加载器
    print('加载数据...')
    train_loader, test_loader, feature_dim, num_users, num_items = create_data_loaders(
        args.train_file,
        args.test_file,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )

    # 初始化模型
    print('初始化模型...')
    model = EcommerceFraudDetector(
        feature_dim=feature_dim,
        num_users=num_users,
        num_items=num_items,
        hidden_dim=args.hidden_dim
    ).to(device)

    # 训练模型
    print('开始训练模型...')
    model = train_model(
        model,
        train_loader,
        None,  # 这里没有验证集，实际应用中应添加
        device,
        epochs=args.epochs,
        lr=args.lr
    )

    # 保存模型
    print(f'保存模型到 {args.model_save_path}')
    torch.save(model.state_dict(), args.model_save_path)

    # 预测
    print('开始预测...')
    predictions = predict(model, test_loader, device, args.output_file)
    print(f'预测完成，结果已保存到 {args.output_file}')


if __name__ == '__main__':
    main()
'''