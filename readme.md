# 代码说明 

## 环境配置 

项目镜像名称为bdc2025

python版本为3.12，其余环境在project文件夹中的requirements.txt中 

## 数据 

仅使用了下发数据进行模型训练

## 预训练模型 

未使用预训练模型

## 算法 

### 整体思路介绍

使用LSTM进行价格预测，在预测结束后计算涨跌幅取出最大和最小的十个股票代码输出

### 网络结构 

网络结构仅为一层LSTM层，一层全连接层进行计算

### 损失函数 

使用损失函数为MSE损失函数

### 数据扩增 

未进行数据扩增

### 模型集成 

仅训练一个LSTM模型，不需要进行模型集成

## 训练流程 

先使用featurework.py进行数据读取和列名转换，在使用train.py进行截取，使用32天的收盘价预测后一天的收盘价创建数据，训练LSTM模型

## 测试流程 

使用test.py读入测试数据，提取最后32天的数据作为预测数据，使用训练完的LSTM模型将每一支股票的新预测结果进行排序得到涨跌幅最大和最小的十个股票代码进行输出
