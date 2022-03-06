![GitHub Dark](https://ibb.co/X7dWvL3)

# NNNS
このレポジトリはNNNS (Numerai Neural Network Studies)という、neural networkを使って[Numerai Tournament](https://numer.ai/tournament)で勝ちたい人たちの勉強会（モブプロ会）の資料を置いておく場所です。

## NNNSの目的
NumeraiはいまだKaggleのようにNN無双になっておらず、NNが活躍できる余地は十二分にあるように思えます。しかし、NNのNumeraiのようなテーブルデータにおける利用のベストプラクティスはまだまだないのが現状です。そのため、最新の技術を含めNNについて学びどんどん実装をしていくことで、Numerai Tournamentで勝てるNNを見つけていくことが本勉強会の狙いです。超えるぞ[XGB](https://www.kaggle.com/code1110/numerai-xgb-baseline)！

### 実際にやること
週１（？）1時間くらいでオンライン開催。モブプロ形式（５名程度参加？）で実装を行う？

- 画面共有 --> Google Meets (会直前にリンク共有)
- モブプロ --> VScode Live Share (Google Meets内でリンク共有)

### 計算環境
[GCP Deep Learning VM Image](https://cloud.google.com/deep-learning-vm)を使う予定。Notebookいるか？ このリポジトリでそのままコードをGit管理する

### 使用データ
Numerai Tournament Legacy Data

https://numer.ai/tournament

### Validation Strategy
使用データの```data_type```に準ずる (要はtime-series split)

### 評価指標
Sharpe ratio...Numerai Diagnostics toolを使った方がいいかも

### 特徴量エンジニアリング
しない（あくまでNN力を高める目的のため）

### NN Framework
[keras](https://keras.io/)を予定（[pytorch](https://pytorch.org/)わからんので教えてくれる人いたらpytorchでも）

## Agenda
- Baseline model (simple MLP)
- [Normalization](https://gaoxiangluo.github.io/2021/08/01/Group-Norm-Batch-Norm-Instance-Norm-which-is-better/)
- [Batch size](https://www.st-hakky-blog.com/entry/2017/11/16/161805)
- [Activation function](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- [Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
- Learning rate scheduler (reducelronplateau, cosine annealing)
- Loss function (MSE, BCE, crossentropy, weighted kappa, corr)
- [Wide & Deep](https://keras.io/examples/structured_data/wide_deep_cross_networks/)
- [Deep & Cross](https://keras.io/examples/structured_data/wide_deep_cross_networks/)
- [Gated residual and variable selection network](https://keras.io/examples/structured_data/classification_with_grn_and_vsn/)
- [Neural decision forest](https://keras.io/examples/structured_data/deep_neural_decision_forests/)
- [Tabnet](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer](https://keras.io/examples/structured_data/tabtransformer/)
- [Denoising Auto Encoder](https://www.kaggle.com/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5)
- [1DCNN](https://www.kaggle.com/c/lish-moa/discussion/202256)
- [evojax](https://github.com/google/evojax)

and more...





