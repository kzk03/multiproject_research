# Focal Loss 論文記述用テンプレート

## 数式

### Binary Cross Entropy (BCE)

$$\text{CE}(p, y) = -y\log(p) - (1-y)\log(1-p)$$

### Focal Loss

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $p_t$: 正解クラスの予測確率
- $\alpha_t$: クラス重み
- $\gamma$: フォーカシングパラメータ

---

## 日本語版

### 短縮版

本研究では、継続・離脱ラベルのクラス不均衡問題に対処するため、損失関数としてFocal Loss [1] を採用した。Focal Lossは、分類が容易なサンプルの損失を低減し、分類が困難なサンプルに学習を集中させることで、不均衡データにおける学習を安定化させる。

### 詳細版

本研究では、損失関数として2値交差エントロピー（Binary Cross Entropy）[2] ではなく、Focal Loss [1] を採用した。Focal Lossは元々、物体検出における前景・背景クラスの極端な不均衡問題に対処するために提案された損失関数である。標準的な交差エントロピー損失 $\text{CE}(p, y) = -y\log(p) - (1-y)\log(1-p)$ に対し、Focal Lossは以下のように定義される：

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

ここで、$p_t$ は正解クラスの予測確率、$\alpha_t$ はクラス重み、$\gamma$ はフォーカシングパラメータである。$(1-p_t)^\gamma$ の項により、高い確信度で正しく分類されたサンプルの損失寄与が低減され、分類が困難なサンプルへの学習が促進される。本研究で対象とするレビュー継続予測タスクでは、継続・離脱ラベルに不均衡が存在するため、Focal Lossの採用により学習の安定化と予測性能の向上を図った。

---

## English Version

### Short Version

To address the class imbalance between continuation and dropout labels, we employed Focal Loss [1] as the loss function. Focal Loss down-weights the loss contribution from easily classified samples and focuses learning on hard examples, thereby stabilizing training on imbalanced data.

### Detailed Version

In this study, we adopted Focal Loss [1] instead of standard Binary Cross Entropy [2] as the loss function. Focal Loss was originally proposed to address the extreme class imbalance between foreground and background classes in object detection. Given the standard cross-entropy loss $\text{CE}(p, y) = -y\log(p) - (1-y)\log(1-p)$, Focal Loss is defined as:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the predicted probability for the ground-truth class, $\alpha_t$ is the class weight, and $\gamma$ is the focusing parameter. The modulating factor $(1-p_t)^\gamma$ reduces the loss contribution from well-classified examples, allowing the model to focus on hard, misclassified samples during training. Since our review continuation prediction task exhibits class imbalance between continuation and dropout labels, we employed Focal Loss to stabilize training and improve prediction performance.

---

## 参考文献 / References

### Focal Loss (原論文)

```bibtex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2980--2988},
  year={2017}
}
```

**引用形式:**
- [1] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.
- arXiv: https://arxiv.org/abs/1708.02002

### Binary Cross Entropy (深層学習教科書)

```bibtex
@book{goodfellow2016deep,
  title={Deep Learning},
  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  year={2016},
  publisher={MIT Press}
}
```

**引用形式:**
- [2] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.

### 情報理論 (クロスエントロピーの原典)

```bibtex
@article{shannon1948mathematical,
  title={A mathematical theory of communication},
  author={Shannon, Claude E},
  journal={The Bell System Technical Journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948}
}
```

### パターン認識 (機械学習教科書)

```bibtex
@book{bishop2006pattern,
  title={Pattern Recognition and Machine Learning},
  author={Bishop, Christopher M},
  year={2006},
  publisher={Springer}
}
```
