# Keras中文官方文档目录

> 整理：张子豪

[TOC]

# 子豪兄Keras视频教程

Github：https://github.com/TommyZihao/zihaokeras

【子豪兄keras】零基础入门Keras



【子豪兄keras】Keras.js与WebDNN：在浏览器中玩转人工智能

https://www.bilibili.com/video/av83160115/

【子豪兄keras】Keras官方文档精读

https://www.bilibili.com/video/av83195032/





# 官方文档主页

中文文档： https://keras.io/zh/ 

英文文档： https://keras.io/ 

Github主页： https://github.com/keras-team/keras 

# Keras常用模型API

Sequential模型：https://keras.io/getting-started/sequential-model-guide/ 

Sequential模型APIhttps://keras.io/models/sequential/

函数式模型： https://keras.io/getting-started/functional-api-guide/ 

函数式模型API：https://keras.io/zh/models/model/

模型输入输出、权重参数：https://keras.io/zh/models/about-keras-models/

# Layers

获取某层权重、配置：https://keras.io/zh/layers/about-keras-layers/

核心网络层Core：Dense、Activation、Dropout、Flatten、Input、Reshape、Permute、RepeatVector、Lambda、ActivityRegularization、Masking、SpatialDropout1D、SpatialDropout2D、SpatialDropout3D

https://keras.io/zh/layers/core/

卷积层：Conv1D、Conv2D、Conv3D、SeparableConv1D、SeparableConv2D、DepthwiseConv2D、Conv2DTranspose、Conv3DTranspose、Cropping1D、Cropping2D、Cropping3D、UpSampling1D、UpSampling2D、UpSampling3D、ZeroPadding1D、ZeroPadding2D、ZeroPadding3D

https://keras.io/zh/layers/convolutional/

池化层：MaxPooling1D、MaxPooling2D、MaxPooling3D、AveragePooling1D、AveragePooling2D、AveragePooling3D、GlobalMaxPooling1D、GlobalAveragePooling1D、GlobalMaxPooling2D、GlobalAveragePooling2D、GlobalMaxPooling3D、GlobalAveragePooling3D

https://keras.io/zh/layers/pooling/

局部连接层：与卷积层相同，只不过卷积核权值不共享。对数据的不同部分应用不同的卷积核。

LocallyConnected1D、LocallyConnected2D

https://keras.io/zh/layers/local/

循环神经网络：RNN、SimpleRNN、GRU、LSTM、ConvLSTM2D、SimpleRNNCell、GRUCell、LSTMCell、cuDNNGRU、CuDNNLSTM

https://keras.io/zh/layers/recurrent/

嵌入层Embedding：https://keras.io/zh/layers/embeddings/

融合层Merge：加减、点乘、平均、最大、拼接Concatenate

https://keras.io/zh/layers/merge/

高级激活函数Advanced Activations：LeakyReLU、PReLU、ELU、ThresholdedReLU、Softmax、ReLU

https://keras.io/zh/layers/advanced-activations/

批归一化BatchNormalization：https://keras.io/zh/layers/normalization/

噪音层：https://keras.io/zh/layers/noise/

层封装器wrappers：TimeDistributed、Bidirectional

https://keras.io/zh/layers/wrappers/

自定义Keras层：https://keras.io/zh/layers/writing-your-own-keras-layers/

# 数据预处理

序列预处理：TimeseriesGenerator、pad_sequences、skipgrams、make_sampling_table

https://keras.io/zh/preprocessing/sequence/

文本预处理：Tokenizer、hashing_trick、one_hot、text_to_word_sequence

https://keras.io/zh/preprocessing/text/

图像预处理：ImageDataGenerator 类及类方法。

https://keras.io/zh/preprocessing/image/

# 损失函数Loss

mean_squared_error、mean_absolute_error、mean_absolute_percentage_error、mean_squared_logarithmic_error、squared_hinge、hinge、categorical_hinge、logcosh

https://keras.io/zh/losses/

# 评价函数Metrics

binary_accuracy、categorical_accuracy、sparse_categorical_accuracy、top_k_categorical_accuracy、sparse_top_k_categorical_accuracy

https://keras.io/zh/metrics/

知乎博客：https://zhuanlan.zhihu.com/p/95293440

# 优化器Optimizers

SGD、RMSprop、Adagrad、Adadelta、Adam、Adamax、Nadam

https://keras.io/zh/optimizers/

# 激活函数

softmax、elu、selu、softplus、softsign、relu、tanh、sigmoid、hard_sigmoid、linear

https://keras.io/zh/activations/

# 回调函数 Callbacks

ModelCheckpoint存储模型、EarlyStopping早停、RemoteMonitor、LearningRateScheduler、TensorBoard可视化、ReduceLROnPlateau学习率减小、CSVLogger、自定义Callbacks函数

https://keras.io/zh/callbacks/

# 常用数据集Datasets

CIFAR10小图像十分类、CIFAR100小图像一百分类、IMDB 电影评论情感分类、路透社新闻主题分类、MNIST手写数字、Fashion-MNIST时尚物品图像、波士顿房价

https://keras.io/zh/datasets/

# 预训练模型Applications

ImageNet图像分类预训练模型：Xception、VGG16、VGG19、ResNet及其变种、Inception及其变种、MobileNet及其变种、DenseNet、NasNet

https://keras.io/zh/applications/

# 后端Backend

Keras是封装在底层张量运算后端之上的模型库，Keras提供的是各种模块，而并不处理底层的张量运算、卷积这些低级操作，这些操作是由后端张量引擎完成的。

Keras目前有三个后端可用，**TensorFlow** 后端，**Theano** 后端，**CNTK** 后端。默认使用的是Tensorflow后端。

据说亚马逊正在开发针对Keras的MXNet后端。

我们可以在keras.json配置文件中切换、配置后端，也可以在代码中直接导入后端，使用后端函数进行张量运算。

比如，`from keras import backend as K`，这里的K相当于`import tensorflow as tf`中的`tf`。

https://keras.io/zh/backend/

# 初始化器Initializers

https://keras.io/zh/initializers/

# 正则化器Regularizers

https://keras.io/zh/regularizers/

# 约束 Constraints

`constraints` 模块的函数允许在优化期间对网络参数设置约束（例如非负性）。

# 神经网路结构与训练过程中指标可视化Visualization

https://keras.io/zh/visualization/

# Scikit-Learn API 的封装器

https://keras.io/zh/scikit-learn-api/

# 常用工具utils

https://keras.io/zh/utils/

# 关于 Github Issues 和 Pull Requests

https://keras.io/zh/contributing/

# 官网案例

可视化VGG卷积核：

https://keras.io/zh/examples/conv_filter_visualization/

https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

Optical character recognition文字识别OCR

使用CNN+RNN进行文字识别，CTC损失函数

https://keras.io/zh/examples/image_ocr/

双向LSTM在IMDB数据集上进行情感分类

https://keras.io/zh/examples/imdb_bidirectional_lstm/



imdb数据集上，使用一维卷积进行文本分类：https://keras.io/examples/imdb_cnn/

stateful=True的循环神经网络：https://keras.io/getting-started/faq/#how-can-i-use-stateful-rnns

猫狗大战图像二分类数据集https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html



# Keras模型部署

苹果Core ML对Keras支持https://developer.apple.com/documentation/coreml

案例：https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/

安卓APP：是不是热狗https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3

Keras.js：https://transcranial.github.io/keras-js/#/

WebDNN：https://mil-tokyo.github.io/webdnn/

Tensorflow-Serving：https://www.tensorflow.org/serving/

Python的Web开发后端Flask部署Keras模型的Rest API：https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

Keras和Flask的进阶教程：https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

# 常见问题解答FAQ

https://keras.io/zh/getting-started/faq/





