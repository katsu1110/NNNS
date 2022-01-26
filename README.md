# NNNS
このレポジトリはNNNS (Numerai Neural Network Studies)という、neural networkを使って[Numerai Tournament](https://numer.ai/tournament)で勝ちたい人たちの勉強会（モブプロ会）の資料を置いておく場所です。

## NNNSの目的
NumeraiはいまだKaggleのようにNN無双になっておらず、NNが活躍できる余地は十二分にあるように思えます。しかし、NNのNumeraiのようなテーブルデータにおける利用のベストプラクティスはまだまだないのが現状です。そのため、最新の技術を含めNNについて学びどんどん実装をしていくことで、Numerai Tournamentで勝てるNNを見つけていくことが本勉強会の狙いです。

### 実際にやること
週１（？）1時間くらいでオンライン開催。モブプロ形式（５名程度参加？）で実装を行う？

### 計算環境
Google colab or kaggle notebookを予定

### 使用データ
Numerai Tournament旧データ（target nomi）

- train data : 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
- validation data : 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'

### Validation Strategy
使用データの```data_type```に準ずる (要はtime-series split)

### 特徴量エンジニアリング
しない（あくまでNN力を高める目的のため）

### NN Framework
tf.kerasを予定（pytorchわからんので教えてくれる人いたらpytorchでも）

## Agenda
- Baseline model (simple MLP)
- [Normalization](https://gaoxiangluo.github.io/2021/08/01/Group-Norm-Batch-Norm-Instance-Norm-which-is-better/)
- [Batch size](https://www.st-hakky-blog.com/entry/2017/11/16/161805)
- [Activation function](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- [Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
- Learning rate scheduler (reducelronplateau, cosine annealing)
- Loss function (MSE, BCE, crossentropy, weighted kappa)
- [Wide & Deep](https://keras.io/examples/structured_data/wide_deep_cross_networks/)
- [Deep & Cross](https://keras.io/examples/structured_data/wide_deep_cross_networks/)
- [Gated residual and variable selection network](https://keras.io/examples/structured_data/classification_with_grn_and_vsn/)
- [Neural decision forest](https://keras.io/examples/structured_data/deep_neural_decision_forests/)
- [Tabnet](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer](https://keras.io/examples/structured_data/tabtransformer/)
- [Use Auto Encoder](https://www.kaggle.com/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5)
- [1DCNN](https://www.kaggle.com/c/lish-moa/discussion/202256)

and more...





