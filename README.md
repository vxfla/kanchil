# kanchil
Kanchil（鼷鹿）是世界上最小的偶蹄目动物，这个开源项目意在**探索**小模型的潜力，通过instruct-tuning、RLHF从零开始赋予模型对齐人类偏好的能力。

GPT4发布后是大规模语言模型的吃鸡赛场，头部瞄准千亿级别，大量力量猛攻百亿级别。我们愿做一股清流，探索下能够对齐人类偏好的小模型，以及如何将千亿百亿规模工作上的经验应用于小模型能力的提升。

p.s. 我们也在chatGLM-6B等模型上进行了继续微调的尝试，可以利用模型原本的能力+新数据集+我们调通的训练脚本进一步开发属于自己的模型。[相关github仓库](https://github.com/27182812/ChatGLM-chinese-insturct)。
# 时间线
- [2023-03-29] 完成1B的MT5-base在[BELLE数据集](https://github.com/LianjiaTech/BELLE)上的微调，其具备了读指令的能力，但是经常胡说八道。开源了[微调后的权重](https://drive.google.com/drive/folders/1aBd_SC9QOl75IVIdAR5i_9Mdpj53vmMY?usp=share_link)

# MT5系列
## MT5-base
MT5-base是一个仅有1B的模型，一张16G的显卡就可以轻松训练、部署，我们测试发现这个规模的模型已经能够通过instruct-tuning获得一定对齐人类偏好的能力。但是受限于模型的规模，其存储的知识并不多，经常会生成事实性/逻辑性有问题的文本。感兴趣的朋友或许可以尝试将一些自己领域的数据用于进一步微调模型，让它成为“领域专家”。
![测试用例](https://s1.ax1x.com/2023/03/29/ppc07wQ.png)

# 模型训练
## 环境准备
参考 requirements.txt

## 训练脚本
目前主要使用huggingface Trainer进行训练，支持deepspeed多卡训练。模型本身的规模比较小，因而也不需要特别的优化，一张16G的显卡可以跑起来。
```
单卡训练： sh train.sh
deepspeed sh deepspeed.sh
```

# 模型推理
```
python chatwithT5.py
```

# 未来工作
- [ ] 优化代码结构，对路径、数据处理等解耦，方便感兴趣的朋友复用我们的代码

- [ ] 在更多不同模型架构、参数量级（不超过6B）的模型上进行instruct-tuning尝试

- [ ] 引入RLHF、检索增强等技术，提高小模型的能力

# 欢迎感兴趣的小伙伴一起探索这个领域！
