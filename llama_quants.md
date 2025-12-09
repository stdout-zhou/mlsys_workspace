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

---

这段函数里还有一个至关重要的实现
``` cpp
static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}
```
它先采用朴素的量化手法, 计算缩放系数
``` math
scale=x_{max} / K
```
`K`代表量化后的最大值
根据这个`scale`可以算出量化后的`l`数组

那么我们可以对`l`计算均方误差来看`scale`选的是否够好

``` math
E = \sum_{i=1}^{n}{(l_i * scale - x_i)^2}
```
这个问题叫最小二乘问题，既然需要最小化`E`,求它对于`scale`的导数, 让导函数为0可得
``` math
scale = \frac {\sum{x_i * l_i}} {\sum{l_i^2}}
```

不过这个做法会依赖一开始朴素算法计算的`scale`, 那个`scale`并不一定是最优的, 所以程序中会有一行`for (int is = -9; is <= 9; ++is)`, 这里会暴力搜`scale`附近的值, 用来作为初始`scale`计算`l`, 然后去对`E`求导
