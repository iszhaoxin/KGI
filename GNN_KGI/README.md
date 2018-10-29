- 将 TransP和TransS变成了线性模型

  效果 : 还是非线性的好

- Merger - 加了一个单纯使用F.average的模型

  效果 : train上升变快, 但是validation效果差?

  不能确定是变差, 很有可能只是因为模型变简单后, 能够更快的收敛而已. 由于没有试验到稳定, 因此不知道哪个的最终效果好. 

- batchsize : 200->1000

  效果 : 一个epoch的速度变慢

- 和简单的shallow神经网络相比, 对trainset的拟合更快, 但是非常同一过拟合, 甚至在一个epoch内便达到了过拟合.

- 在不更新的条件下得到的 Validation accuracy 是: 0.575

  在不更新的条件下得到的 train accuracy 是: 0.55

- ​