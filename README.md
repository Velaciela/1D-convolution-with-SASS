# 1D-convolution-with-SASS

本项目使用MaxAs提供的汇编器编写GPU一维卷积算法
有16点和1024点两种实现，SASS代码在目录下，两个VS2013工程在文件夹中
实验环境：WIN7 + VS2013 + CUDA 6.5 + MaxAs(sass代码已注入cubin，运行不需要MaxAs)

---

16点卷积和1024点卷积恰好可划分在访存密集和计算密集这两个类型中，是比较好的练手项目。
1024点卷积在Maxwell架构下达到硬件75%的峰值算力占用。
|结果|达到硬件峰值计算能力的74% （未达到90%）|
|:---:|:---:|
|数据量| 输入X+输出Y = 2*2097152 float = 2*2097152*4bytes/1024/1024 = 16MB|
|kernel时间|1.4ms|
|带宽占用|16MB / 1.4ms = 11.42GB/s |
|kernel算力|2097152_X输入*1024_H卷积核*2_乘加 / 1.4ms = 4.295 GigaFloatOp / 0.0014s = 3067.86 GFLOPS|
|硬件算力|1.25GHz * 1664cores * 2 = 4160 GFLOPS| 

---

由于github中的markdown对latex公式支持不太好
文档请移步：https://www.zybuluo.com/Velaciela/note/541967
访问密码：Velaciela

---


MaxAs是一个开源的Maxwell架构GPU汇编器： [Github链接][1]

  [1]: https://github.com/NervanaSystems/maxas
