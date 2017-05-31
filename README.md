# 1D-convolution-with-SASS

本项目使用MaxAs提供的汇编器编写GPU一维卷积算法

有16点和1024点卷积核两种实现，SASS代码在目录下，两个VS2013工程在文件夹中

实验环境：WIN7 + VS2013 + CUDA 6.5 + MaxAs (SASS代码已注入cubin，运行不需要MaxAs)

---

16点卷积和1024点卷积恰好可划分在访存密集和计算密集这两个类型中，是比较好的练手项目。

1024点卷积在Maxwell架构下达到硬件75%的峰值算力，还在找未达到90%以上的原因。


---

GitHub中的Markdown对Latex公式支持不太好

文档请移步：[一维卷积的SASS实现][2]

访问密码：Velaciela

---


MaxAs是一个开源的Maxwell架构GPU汇编器： [Github链接][1]

  [1]: https://github.com/NervanaSystems/maxas
  [2]: https://www.zybuluo.com/Velaciela/note/541967
