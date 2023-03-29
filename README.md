# kanchil
Kanchil（鼷鹿）是世界上最小的偶蹄目动物，这个开源项目意在**探索**小模型是否也能具备和人类偏好对齐的能力。

GPT4发布后是大规模语言模型的吃鸡赛场，头部瞄准千亿级别，大量力量猛攻百亿级别。我们愿做一股清流，探索下能够对齐人类偏好的最小模型规模，以及如何将世界上千亿百亿规模工作上的经验应用于小模型能力的提升。

# 时间线
- [2023-03-29] 完成1B的MT5-base在[BELLE数据集](https://github.com/LianjiaTech/BELLE)上的微调，其具备了读指令的能力，但是经常胡说八道。开源了[微调后的权重](https://drive.google.com/file/d/125hjpeS98qum5817XMPp7nY8L19aiOvJ/view?usp=sharing)

# 1B MT5效果
已经能够读懂人类指令，在部分问题上能够做出好的回答，但整体而言模型的基础能力不足。
![测试用例](https://imgse.com/i/ppc07wQ)
# 模型训练
## 环境准备
参考 requirements.txt

## 训练脚本
```
单卡训练： sh train.sh
deepspeed sh deepspeed.sh
```

# 模型推理
```
python chatwithT5.py
```

# 未来工作
- [ ] 优化代码结构，对路径、数据处理等解耦

- [ ] 探索新的规模、训练方式等

# 欢迎感兴趣的小伙伴一起探索这个领域！
