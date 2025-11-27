# ONNX
onnx是把torch模型用protobuf序列化后的产物，通过跨平台和语言的onnx runtime api进行推理调用。
优势:
* 训练的时候torch维护的是一个动态计算图，为了反向传播，会在layer间缓存很多状态，而推理不需要，因为没有反向计算。
* 因为推理没有反向计算，所以不用考虑梯度怎么传播，一些算子可以融合计算。
* 静态计算图的优势，可以提前分配好显存

# TensorRT
英伟达提供的在N卡上推理的runtime api。
相比ORT进一步优化:
* 算法上，TensorRT维护了几十种矩阵乘法算法，用于特定处理不同shape的矩阵输入。甚至每种算子可能也有几十种实现，针对不同的N卡定向优化。
* 参数量化，用更低精度的数值类型代替高精度数值类型参与计算。
* 其他手法，比如更激进的算子融合

---

TensorRT用的时候会有一个compile过程，这里其实是根据模型参数和具体硬件类型来生成量化的参数，以及优化计算图等等，称为`build enginee`。
所以部署的时候可以做一次warmup，把enginee缓存起来。

TensorRT有很多不支持的cuda算子，工程上一般会选择ORT + TensorRT的方式部署模型，推理前框架会对计算图做划分，看哪些算子让TensorRT计算，
哪些给ORT计算。

# Kernel fusion
推理时做算子融合主要收益是减少launch kernel的次数，以及减少gpu访存的次数。

推理框架会自动帮忙做些fusion，一般是基于一些固定的pattern，比如连续出现conv + batch norm。

但有些场景, 比如conv + conv, 要对卷积参数reshape, 框架没有聪明到知道这两个conv之间能做数学合并, 这时候还是依赖导出onnx时写代码手动合并。

目前, 一些AI编译器能对计算图做扫描, 通过图论分析技术对element-wise的算子做合并。

更为复杂的fusion, 比如任意算子合并, 这部分我暂时还没从网上找到有什么解决办法。

# CUDA kernel
1. spconv
refer from:
* https://zhuanlan.zhihu.com/p/698314399
* https://www.mdpi.com/1424-8220/18/10/3337

输入大部分是zero value, 导致在输入上运行卷积核时有大量无效运算。

假设输入shape是(h, w, 1), 代表高, 宽, 通道。卷积核是3 * 3, 输出shape是(h-2, w-2, 1)。

它的做法是构建一个叫rulebook的东西，他会从输入里拿所有的非0值, 然后构建输入idx到输出idx的映射，再到卷积核offset的映射。

前一个映射比较好懂，就是看输入的激活值被映射到输出的哪个位置。后一个映射是说卷积时，卷积核的哪个位置起作用。那么3 * 3的kernel最是9种offset.

实际计算的时候每个offset读出来，然后gather激活点，gemm做矩乘，最后scatter到output上。


