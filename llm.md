# Transformer
使用监督学习的网络结构。
把句子分割成若干词语(token), 给每个token一个embedding, 同时因为token的位置关系也很重要, 所以要叠加position embedding。
维护三个权重矩阵Q, K, V, 每个词token和Q, K, V做矩乘得到O_Q, O_K, O_V。
用当前token的O_Q去和所有token的O_K做点积运算, 并归一化, 会得到一个新的向量, 称它为attention score, attention score越大, 代表和当前token关系越紧密。
用attention score里的每个标量去和所有token的O_V做乘法再相加, 得到的就是self-attention学习到的信息。
上面的所有过程称为encoder。

把encoder的输出过一个FFN全连接层网络, 并归一化, 生成一个目标字典长度的新的向量, 向量中第i个数值代表下一个token是目标字典第i个单词的概率。
这个过程称为decoder。

把decoder的结果和标签算loss, 然后反向计算, 就能更新权重矩阵Q, K, V。

有时候Q, K, V不止一个, 多个Q, K, V在encoder时分别计算，然后叠加结果, 这种算法称作多头注意力机制。


# Inference优化手法
## K-V cache
llm每次生成下一个token(forward)时, 都必须用到之前的整个句子的信息, 那么之前forward的token的k, v可以缓存下来。
并且由于推理时每个token只会关注过去, 而不关注未来, 在以后的token生成中, 之前生成的token都无需再重新计算attention score。
