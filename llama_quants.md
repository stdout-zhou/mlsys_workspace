# q4_0
表示用4bit存储一个float32, 0是一种量化策略

核心实现
``` cpp
static void quantize_row_q4_0_impl(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t n_per_row, const float * quant_weights) {
    static_assert(QK4_0 == 32, "QK4_0 must be 32");

    if (!quant_weights) {
        quantize_row_q4_0_ref(x, y, n_per_row);
        return;
    }

    float weight[QK4_0];
    int8_t L[QK4_0];

    float sum_x2 = 0;
    for (int j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
    float sigma2 = sum_x2/n_per_row;

    const int64_t nb = n_per_row/QK4_0;
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + QK4_0 * ib;
        const float * qw = quant_weights + QK4_0 * ib;
        for (int j = 0; j < QK4_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
        y[ib].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < 16; ++j) {
            y[ib].qs[j] = L[j] | (L[j+16] << 4);
        }
    }
}
```
代码讲解

* 对一行数据按照`block_size=32`做分块
* 根据参数传入的`qw`和这行的平方均值`sigma`对输入权重做平滑, 这样可以防止权重消失并在当前block上融合全局信息
* 调用`make_qx_quants`在当前block上计算缩放系数
* 记录当前block的缩放系数和量化后权重

--- 

Q1: `y[ib].qs[j] = L[j] | (L[j+16] << 4)`这行代码在做什么？

A1: 把量化后的两个结果用一个byte表示, 因为c++最小寻址单元是byte, 不满1 byte编译器可能会自动做padding

Q2: 为什么要分block计算?

A2: 因为全局参数的最大和最小值差距可能非常大, 所有参数一起量化明显会导致量化后的精度下降。但分block量化显然也有一个副作用, 每个block都必须存储一个fp32的缩放系数。
