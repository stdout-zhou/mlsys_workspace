# Data transfer

## pin_memory
[pytorch tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

数据从cpu RAM搬运到gpu VRAM时，不会经过cpu，可以通过PCle直接拷贝。但这个过程需要tensor的物理内存地址不发生改变，即拷贝时不能有memory page被交换到硬盘上。

在cpu上创建tensor时，通过指定 pin_memory=True 让tensor使用锁页内存, H2D时不会带来cpu把数据从分页内存拷贝到锁页内存的开销
。
gpu上不存在锁页内存和分页内存一说，因为显存不会和硬盘进行内存交换。
```python
def benchmark_pin_memory(pin_memory: bool):
    import time

    import torch

    cpu_tensor = torch.randn(1024 ** 3, device="cpu", pin_memory=pin_memory)
    gpu_tensor = torch.randn(1024 ** 3, device="cuda:0", pin_memory=False)

    # cpu transter to gpu
    start = time.time()
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print(f"pin_memory: {pin_memory} H2D time cost: {end -  start}")

    # gpu transter to cpu
    start = time.time()
    cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print(f"pin_memory: {pin_memory} D2H time cost: {end - start}")


benchmark_pin_memory(pin_memory=False)
benchmark_pin_memory(pin_memory=True)

# Bechmark result:
# pin_memory: False H2D time cost: 0.8497984409332275
# pin_memory: False D2H time cost: 0.33328771591186523

# pin_memory: True H2D time cost: 0.4279017448425293
# pin_memory: True D2H time cost: 0.08166313171386719
```

发现使用 pin_memory 后数据拷贝速度快了很多，但 D2H 的时间和 H2D 并不一致，这是因为程序的第一个cuda操作会有额外开销，我们多增加一个synchronize来观察耗时。

```python
def benchmark_pin_memory(pin_memory: bool):
    import time

    import torch

    cpu_tensor = torch.randn(1024 ** 3, device="cpu", pin_memory=pin_memory)
    gpu_tensor = torch.randn(1024 ** 3, device="cuda:0", pin_memory=False)

    # wram up
    torch.cuda.synchronize()

    # cpu transter to gpu
    start = time.time()
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print(f"pin_memory: {pin_memory} H2D time cost: {end -  start}")

    # gpu transter to cpu
    start = time.time()
    cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()
    end = time.time()
    print(f"pin_memory: {pin_memory} D2H time cost: {end - start}")

# Bechmark result:
# pin_memory: False H2D time cost: 0.4834115505218506
# pin_memory: False D2H time cost: 0.32852792739868164

# pin_memory: True H2D time cost: 0.09655046463012695
# pin_memory: True D2H time cost: 0.08164072036743164
PCle是一条可以双向传输数据的总线， D2H和H2D耗时相近。
```

## numa
[pytorch tutorial](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#numactl)

全称`Non-Uniform Memory Access`，中文非一致性内存访问。

传统SMP architecture下，一条总线会串连所有cpu和RAM。cpu核心scale以后，和RAM的通信受到总线数量的限制。

`numa`把多个cpu core和一部分RAM划分为一个`node`, node内部的数据通信很快，node之间的通信速度取决于distance。

![numa architecture](image.png)

`numactl --hardware` show cpu core绑定到哪个node
```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
node 0 size: 1031257 MB
node 0 free: 542181 MB
node 1 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
node 1 size: 1032165 MB
node 1 free: 660070 MB
node distances:
node   0   1 
  0:  10  21 
  1:  21  10 
```

`nvida-smi topo -m` show gpu和cpu的topo
```
        GPU0    GPU1    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7  NIC8     NIC9    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NODE    NODE    NODE    NODE    NODE    PIX     SYS     SYS   SYS      SYS     0-47,96-143     0               N/A
GPU1    NV18     X      SYS     SYS     SYS     SYS     SYS     SYS     PIX     NODE  NODE     NODE    48-95,144-191   1               N/A
NIC0    NODE    SYS      X      PIX     NODE    NODE    NODE    NODE    SYS     SYS   SYS      SYS
NIC1    NODE    SYS     PIX      X      NODE    NODE    NODE    NODE    SYS     SYS   SYS      SYS
NIC2    NODE    SYS     NODE    NODE     X      NODE    NODE    NODE    SYS     SYS   SYS      SYS
NIC3    NODE    SYS     NODE    NODE    NODE     X      NODE    NODE    SYS     SYS   SYS      SYS
NIC4    NODE    SYS     NODE    NODE    NODE    NODE     X      NODE    SYS     SYS   SYS      SYS
NIC5    PIX     SYS     NODE    NODE    NODE    NODE    NODE     X      SYS     SYS   SYS      SYS
NIC6    SYS     PIX     SYS     SYS     SYS     SYS     SYS     SYS      X      NODE  NODE     NODE
NIC7    SYS     NODE    SYS     SYS     SYS     SYS     SYS     SYS     NODE     X    NODE     NODE
NIC8    SYS     NODE    SYS     SYS     SYS     SYS     SYS     SYS     NODE    NODE   X       NODE
NIC9    SYS     NODE    SYS     SYS     SYS     SYS     SYS     SYS     NODE    NODE  NODE      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9
```
在这台机器上`gpu0`的cpu affinity是`cpu 0-47,96-143`，这正对应了`node0`

用`numactl --cpunodebind N --membind N python <script>`测试一下不同node和`gpu0`的数据拷贝速度
```
> numactl --cpunodebind=0 --membind=0 python3 toy.py
pin_memory: True H2D time cost: 0.08621597290039062
pin_memory: True D2H time cost: 0.08167719841003418

> numactl --cpunodebind=1 --membind=1 python3 toy.py
pin_memory: True H2D time cost: 0.09686589241027832
pin_memory: True D2H time cost: 0.08164668083190918
```
`gpu0`和`node0`之间数据拷贝更快。
