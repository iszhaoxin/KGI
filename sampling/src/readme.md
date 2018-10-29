#### 1. Task

把整个图分割成两个 sub-graph



#### 2. Statistical information 

这里对于一个图, 构建两个性质的统计工具. 

-  Degree distribution
- clustering coefficient distribution

最后的结果需要对生成的子图和原图的性质进行对比.



#### 3. Method

- 将图信息编入 nx class 中


- 按照`特定采样策略`采样出三个独立的点集
  - subNode1 : subG1 特有node
  - subNode2 : subG2 特有node
  - shareNode ： 两个图共有node
- 利用分割后的点集进行子图构建.



#### 4. Sampling strategies

##### 4.1 Uniform random sampling

按照均匀分布的概率随机抽取node.

##### 4.2 Weighted random sampling

按照点的度数作为权重进行sampling

- $W(n)$ 正比于 $1/D(n)$
- $W(n)$ 正比于 $1/\log(D(n))$

需要对这两个进行分析, 看那一个在 sampling 后还能和原来的图保有一样的统计特性

##### 4.3 Cohisive sampling

凝聚子团sampling.

