文件组成如下：  
data_loader：保存所有数据（由于仅用于示例演示，所以数据来源为ai生成，不具备真实性）  
main：主要运行文件，用于微调模型的生成和合并  
inference：用于使用微调模型，输入一条评论信息以获取评价  

1.5b版本可以在4070上运行  
  
运行方法：先运行main，完成之后运行inference  

模型下载地址：  
7b：https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main  
1.5b：https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/tree/main
