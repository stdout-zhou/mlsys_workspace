> **_NOTE:_**  训练常用手法

# DP
数据并行(Data Parallelism)

多机多卡训练时, 可以把一个batch的数据均分给不同的卡。称为map。

每张卡独立计算loss后进行forward, 利用MPI集合通信原语同步梯度, 这样单张卡就能收到其他卡的梯度信息, 求均值后更新模型参数, 实现多卡数据并行。称为reduce。

DP是最简单, 也是使用最多的多卡训练手法。

# TP
张量并行(Tensor Parallelism)

DP是复制模型，切分数据。

模型太大, 单卡显存存不下模型参数时, 可以用TP。

把权重矩阵按列拆分, 计算完结果后多卡集合通信把结果拼接。

把权重矩阵按行拆分, 此时batch要按列切分, 计算完结果后多卡集合通信把结果相加。

TP每次forward和backward都要集合通信, 通信的开销很大。

# PP
流水线并行(Pipeline Parallelism)

把不同的layer切分到多卡上, 一张卡forward完一个batch后, 就可以把输出发给下一张卡, 然后立即处理下一个batch。

PP要考虑怎么不让每一个环节不成为整个pipeline的bottleneck。
