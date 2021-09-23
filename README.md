# 说明
模式识别作业pytorch版百行baseline，仅供参考

# 使用方法
安装Pytorch > 1.2，torchvision，tqdm（可有可无）

修改baseline.py中的路径
```bash
python baseline.py
```

## 优化：
* 划分验证集，取验证集最高时的测试集结果作为最终结果
* 进一步，做k折验证
* 数据增强
* 正负样本平衡
* 换其他模型
* 修改优化器，换成Adam等
* 加入更多的评价指标，比如F1-score
* matplotlib画图
