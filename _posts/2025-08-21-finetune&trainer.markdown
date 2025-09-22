---
layout: post
title: "fine-tune & Trainer"
author: "Brahmsky"
date: 2025-08-21 18:00:00 +0800  # 文章发布日期和时间（可选，但推荐）
categories: [深度学习, PyTorch]      # 文章分类（可以是多个）
tags: [Transformer, 注意力机制, NLP] # 文章标签（可以是多个）
catalog: true                       # 是否显示目录 (根据你的主题)
header-style: text                  # 头部样式 (根据你的主题)
---

这一节是重点，整体NLP编程范式和trainer的使用方法，但是能讲的不多。重点看代码吧。

加载bert-base-uncased，对imdb做微调。raw_datasets = load_dataset('imdb')后，
**`raw_datasets` (即 `DatasetDict` 对象) 有三个键：`train` 和 `test`和`unsuperivsed`。**`raw_datasets['train']` 以及另外两个，都是Dataset类，都有 `text` 键，和一个 `label` 键。
```python
import torch  
from datasets import load_dataset  
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer  
from transformers import DataCollatorWithPadding  
import evaluate # Hugging Face 提供的评估指标库  
import numpy as np

raw_datasets = load_dataset('imdb') # DatasetDict({train/test/unsupervised:Dataset({features:['text','label'],num_rows:25000})
tr_set = raw_dataset['train']
va_set = raw_dataset['test'] # Dataset({features:['text','label'],num_rows:25000})

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # sample tokenizer(& trunctate)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)#batch packer & batch padder
# 这里pad的idx和方向之类的其实也是定义在tokenizer里面的
def sample_tokenize(sample):
	return tokenizer(sample['text'], truncation=True, max_length=512) # return_tensor='np'

metric_arruracy = evaluate.load("accuracy") # 从hf的evaluate库加载用于计算准确率的评估器对象
def compute_metrics(eval_pred):# 创建一个评估函数作为trainer的回调函数，于每epoch结束后
	"""
	args:
		eval_pred: Trainer自动将logits和labels打包成的tuple(logits, IDs)
	"""
	logits, labels = eval_pred # logits和labels都是 numpy.ndarray !!!
	preds = np.argmax(logits, axis=-1) # 如果用torch.argmax就会报错
	
	res = {} # 你纵使可以直接return metric_accuracy.compute()，但这是compute_metrics最普遍的写法
	res.update(metric_accuracy.compute(predictions=preds, references=labels))# pred_labels, labels
	return res
```
**注意，sample_tokenizer中的tokenizer()的return_tensor是默认值np，不是torch。**

全局中需要写的东西，数据集、sample tokenizer、data collator、compute metrics。这个涉及到Python `multiprocessing` 在 Windows 上的“生成”（Spawn）机制。`if __name__ == '__main__':` 块用于封装那些**只应该在主进程直接运行时执行一次**的代码，所以首先，用于启动子进程的代码一定不能写在全局，要不然当子进程重新导入主脚本时会再次执行，无限递归地启动进程，引发 `RuntimeError`。为了子进程能初始化和访问主脚本共享的工具和数据，顺利并行处理，首先sample_tokenize及其依赖tokenizer肯定是要写在主脚本的。至于datasets、metric、data collator，子进程完全可以拥有自己的实例，这是安全的，函数在没有调用之前也只是占位的代码而已。

关于metric。metric是啥？是`evaluate.Metric` 类的实例，一个 Python 对象，封装了计算准确率的逻辑。compute_metrics函数计算我们关心的指标（如准确率，召回率，F1），并以字典形式返回。`metric` 对象在脚本的全局作用域加载一次即可。
    - **值/内容：** 本身不存储准确率值，而是提供了一个 `.compute()` 方法来计算准确率。
    - **用法：** 通过调用 `metric.compute(...)` 来实际计算准确率。
metric的输入是什么？Trainer 在每一个epoch后回调评估函数时，会将eval_datasets中每个batch的模型预测logits和labels**打包成`Tuple(ndarray)`** eval_pred 传进去。务必注意logits和labels的**ndarray形式**，因为这涉及到你这个函数怎么写：写`preds = np.argmax(logits, axis=-1)`而不是`torch.argmax(logits, dim=-1)`，要不然报TypeError。Trainer是想和所有深度学习框架适配的，当然会指定numpy.ndarray。Trainer在内部将GPU上的Tensor**收集并转移到CPU**后，会统一转换成NumPy数组，再交给compute_metrics函数，可以和其他框架无缝衔接。

以及，compute里面传入的是predictions和references，注意有复数，传单数的话那是另一个意思了，会报错。
```python
if __name__=="__main__":
	
	tokenized_tr_set = tr_set.map(sample_tokenize, batched=True, num_proc=4) # 可添加num_proc指定并行进程数
	tokenized_va_set = va_set.map(sample_tokenize, batched=True, num_proc=4)#Dataset({features:['text','label','input_ids','token_type_ids','attention_mask'],num_rows: 25000})
	tokenized_tr_set = tokenized_tr_set.remove_columns(['text']).rename_column("label", "labels").with_format("torch") # 很常见的实践！移除文本列，标签列重命名，转换格式
	tokenized_va_set = tokenized_va_set.remove_columns(['text']).rename_column("label", "labels").with_format("torch")
	
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 自动为MODEL_NAME这个模型添加/替换分类头
	
	training_args = TrainingArguments(
		output_dir="models/my_awesome_model",  
		num_train_epochs=3,  
		per_device_train_batch_size=40,  
		gradient_accumulation_steps=1,  
		per_device_eval_batch_size=16,  
		warmup_steps=500,  
		weight_decay=1e-2,  
		logging_dir="./logs",  
		logging_steps=10,  
		eval_strategy="epoch",  
		save_strategy="epoch",  
		load_best_model_at_end=True, # 把表现最好的那个checkpoint的内容复制到顶层输出目录
		fp16=True,  
		dataloader_num_workers=4,
	)
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_tr_set,
		eval_dataset=tokenized_va_set,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)
	
	print("---------------start training---------------")
	trainer.train()
	print("---------------training complete---------------")
	
	eval_res = trainer.evaluate()
	print(f"Accuracy on validation data: {eval_res['accuracy']: .4f}")
```
几个小点：
处理数据集，`tok_va_set = tok_va_set.remove_columns(['text']).rename_column("label", "labels").with_format("torch")`中，`.with_format("torch")`是`datasets`库提供的标准化方法，会统一将数值字段（如`input_ids`、`attention_mask`）转为PyTorch张量，并保留数据集的元信息（如列名、类型），与`datasets`生态无缝集成（如支持`DataCollator`、`Trainer`等）。
而`ToTensor()`通常来自`torchvision.transforms`，设计用于图像数据转换，若强行用于文本数据集，可能导致：丢失列名信息，后续处理易出错；需手动指定转换字段，代码冗余。若原始数据包含非数值字段（如类别标签），`ToTensor()`可能引发类型错误，而`.with_format("torch")`会跳过非数值列。

AutoModelForSequenceClassification中的**num_labels=2 是关键**。相当于加载通用的BERT模型，但是把它顶部的、用于MLM任务的头扔掉，然后随机初始化一个新的、输出维度为2的线性分类头。

Trainer 的参数，分成log、epoch、batch、lr、strategy、加速以及是否load_best_model_at_end等等。

关于pad和trunctation：
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def sample_tokenize(sample):
	return tokenizer(sample['text'], truncation=True, max_length=512)
	
if __name__=='__main__':
	tokenized_tr_set = tr_set.map(sample_tokenize, batched=True, num_proc=4)
```
这个写法中，由于`return tokenizer()`中没有写`padding="max_length"`，也就是我们没有采用手动指定padding长度，于是主要任务就是encode + truncation，没有padding。那么padding实际上交给谁呢？
```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if __name__=='__main__':
	trainer = Trainer(……, data_collator=data_collator, ……)
```
这里才是padding的地方。
这个是最佳实践。想手动指定padding长度`padding="max_length"`的话，写进`sample_tokenzie`里面就可以，这样在最开始处理数据集的时候就已经pad好了，trainer里面就不用写`data_collator`。这里依然需要传入tokenizer，因为里面还存着`pad_token_id`和`padding_side`,`model_input_names`等，需要指导padding工作（本质上是一个**依赖注入Dependency Injection**的过程）。
有点类似于nn.Module里头的buffer，但是实际上这个是存在一个模型的配置文件（例如 `config.json`）中的。
