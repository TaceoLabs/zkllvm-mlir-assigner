# zkML TestSuite

This document serves two purposes.

- It explains how to run tests for the zkML pipeline.
- It keeps track of the supported ONNX operations, along with the current pinned
  version of ONNX-MLIR.

## TLDR: How run Test Suite

In the home directory of this repository, after building the project, run the
following command for a quick check:

```
python tests/run.py --fast
```

If you have some time run the same script without the `--fast` flag (in the home
directory, after building) and grab a coffee, this takes some time:

```
python tests/run.py
```

## Folder Structure

Inside the `/tests` folder (where you found this README.md) is a python script
and two subfolders.

- `/Models` Inside this folder are test cases for pre-trained models. At the
  moment, there are tests for two MNIST models, whereas one is a CNN.
- `/Ops` Inside this folder you can find specific test casses for supported ONNX
  operations.
- `run.py` A python script that executes the tests and gathers the information
  of a run of the test suite. Add the additional flag `--fast` to run only the
  tests in the `/Ops` folder. If you omit the flag, it will also run the tests
  in the `/Models` folder. Testing the pre-trained models will take some time.
- `README.md` This README.

### Quick Check

In contrast to the pre-trained existing models in the `/Models` folder, the
"models" in the `/Ops` folder test a single ONNX operation. For every supported
ONNX operation you can find a subfolder that holds compiled ONNX models that
only consist of the specific operation. E.g., the file `/Ops/Add/AddSimple.mlir`
is compiled from an "ONNX model" only consisting of a single
[ONNX Add operation](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add).

Some of the more involved operations have additional attributes to define their
semantics. In such cases, you can find more than a single test case in the
folder.

The rest of the document exhaustively defines all supported ONNX operations with
known limitations.

# Supported ONNX Operations

The current LLVM version of zkLLVM (`zkllcm-circifier`) is based on 16.0.0-rc4.
Therefore we use onnx-mlir
[at this commit](https://github.com/onnx/onnx-mlir/tree/a04f518c1b0b8e4971d554c399bb54efc00b81db)
as it incorporated with this version of LLVM. This document keeps track of the
supported ONNX operations of the zkML frontend and its limitations.

For further information on the operations, see the
[ONNX documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

| Op                            | zkML Support | ONNX-MLIR support  | Limitations                                                                                                                                        |
| ----------------------------- | :----------: | :----------------: | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Abs**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Acos**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Acosh**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Adagrad**                   |     :x:      |        :x:         |                                                                                                                                                    |
| **Adam**                      |     :x:      |        :x:         |                                                                                                                                                    |
| **Add**                       |     :x:      | :white_check_mark: | No support for short integers.                                                                                                                     |
| **And**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ArgMax**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ArgMin**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ArrayFeatureExtractor**     |     :x:      |        :x:         |                                                                                                                                                    |
| **Asin**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Asinh**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Atan**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Atanh**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **AveragePool**               |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **BatchNormalization**        |     :x:      | :white_check_mark: | Training not supported.                                                                                                                            |
| **Bernoulli**                 |     :x:      |        :x:         |                                                                                                                                                    |
| **Binarizer**                 |     :x:      |        :x:         |                                                                                                                                                    |
| **BitShift**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **BitwiseAnd**                |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **BitwiseNot**                |     :x:      |        :x:         |                                                                                                                                                    |
| **BitwiseOr**                 |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **BitwiseXor**                |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **BlackmanWindow**            |     :x:      |        :x:         |                                                                                                                                                    |
| **Cast**                      |     :x:      | :white_check_mark: | Cast only between float and double types. Only ppc63le and MacOS platforms support float16.                                                        |
| **CastLike**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **CastMap**                   |     :x:      |        :x:         |                                                                                                                                                    |
| **CategoryMapper**            |     :x:      |        :x:         |                                                                                                                                                    |
| **Ceil**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Celu**                      |     :x:      |        :x:         |                                                                                                                                                    |
| **CenterCropPad**             |     :x:      |        :x:         |                                                                                                                                                    |
| **Clip**                      |     :x:      | :white_check_mark: | No support for short integers.                                                                                                                     |
| **Col2Im**                    |     :x:      |        :x:         |                                                                                                                                                    |
| **Compress**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Concat**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ConcatFromSequence**        |     :x:      |        :x:         |                                                                                                                                                    |
| **Constant**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ConstantOfShape**           |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Conv**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ConvInteger**               |     :x:      |        :x:         |                                                                                                                                                    |
| **ConvTranspose**             |     :x:      | :white_check_mark: | Unknown dimension in spatial dimensions (such as H and W) not supported.                                                                           |
| **Cos**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Cosh**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **CumSum**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **DFT**                       |     :x:      |        :x:         |                                                                                                                                                    |
| **DeformConv**                |     :x:      |        :x:         |                                                                                                                                                    |
| **DepthToSpace**              |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **DequantizeLinear**          |     :x:      | :white_check_mark: | Only support for per-tensor or layer dequantization. No support for per-axis dequantization.                                                       |
| **Det**                       |     :x:      |        :x:         |                                                                                                                                                    |
| **DictVectorizer**            |     :x:      |        :x:         |                                                                                                                                                    |
| **Div**                       |     :x:      | :white_check_mark: | No support for short integers.                                                                                                                     |
| **Dropout**                   |     :x:      | :white_check_mark: | Does not support masked and training.                                                                                                              |
| **DynamicQuantizeLinear**     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Einsum**                    |     :x:      | :white_check_mark: | Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 31.               |
| **Elu**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Equal**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Erf**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Exp**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Expand**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **EyeLike**                   |     :x:      |        :x:         |                                                                                                                                                    |
| **FeatureVectorizer**         |     :x:      |        :x:         |                                                                                                                                                    |
| **Flatten**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Floor**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GRU**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Gather**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GatherElements**            |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GatherND**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Gemm**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GlobalAveragePool**         |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GlobalLpPool**              |     :x:      |        :x:         |                                                                                                                                                    |
| **GlobalMaxPool**             |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Gradient**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **Greater**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GreaterOrEqual**            |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **GridSample**                |     :x:      |        :x:         |                                                                                                                                                    |
| **GroupNormalization**        |     :x:      |        :x:         |                                                                                                                                                    |
| **HammingWindow**             |     :x:      |        :x:         |                                                                                                                                                    |
| **HannWindow**                |     :x:      |        :x:         |                                                                                                                                                    |
| **HardSigmoid**               |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **HardSwish**                 |     :x:      |        :x:         |                                                                                                                                                    |
| **Hardmax**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Identity**                  |     :x:      | :white_check_mark: | Sequence identity not supported.                                                                                                                   |
| **If**                        |     :x:      | :white_check_mark: | Sequence and Optional outputs are not supported.                                                                                                   |
| **Imputer**                   |     :x:      |        :x:         |                                                                                                                                                    |
| **InstanceNormalization**     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **IsInf**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **IsNaN**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LRN**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LSTM**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LabelEncoder**              |     :x:      |        :x:         |                                                                                                                                                    |
| **LayerNormalization**        |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LeakyRelu**                 |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Less**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LessOrEqual**               |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LinearClassifier**          |     :x:      |        :x:         |                                                                                                                                                    |
| **LinearRegressor**           |     :x:      |        :x:         |                                                                                                                                                    |
| **Log**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LogSoftmax**                |     :x:      |                    | :white_check_mark: Axis -1, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13.                                               |
| **Loop**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **LpNormalization**           |     :x:      |        :x:         |                                                                                                                                                    |
| **LpPool**                    |     :x:      |        :x:         |                                                                                                                                                    |
| **MatMul**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **MatMulInteger**             |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Max**                       |     :x:      | :white_check_mark: | No support for unsigned int. Only ppc63le and MacOS platforms support float16.                                                                     |
| **MaxPool**                   |     :x:      | :white_check_mark: | Does not support argmax and short ints. Support single output only.                                                                                |
| **MaxRoiPool**                |     :x:      |        :x:         |                                                                                                                                                    |
| **MaxUnpool**                 |     :x:      |        :x:         |                                                                                                                                                    |
| **Mean**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **MeanVarianceNormalization** |     :x:      |        :x:         |                                                                                                                                                    |
| **MelWeightMatrix**           |     :x:      |        :x:         |                                                                                                                                                    |
| **Min**                       |     :x:      | :white_check_mark: | Does not support unsigned numbers. Only ppc63le and MacOS platforms support float16.                                                               |
| **Mish**                      |     :x:      |        :x:         |                                                                                                                                                    |
| **Mod**                       |     :x:      | :white_check_mark: | Support float and double only. Only ppc63le and MacOS platforms support float16.                                                                   |
| **Momentum**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **Mul**                       |     :x:      | :white_check_mark: | Does not support short integers.                                                                                                                   |
| **Multinomial**               |     :x:      |        :x:         |                                                                                                                                                    |
| **Neg**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **NegativeLogLikelihoodLoss** |     :x:      |        :x:         |                                                                                                                                                    |
| **NonMaxSuppression**         |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **NonZero**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Normalizer**                |     :x:      |        :x:         |                                                                                                                                                    |
| **Not**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **OneHot**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **OneHotEncoder**             |     :x:      |        :x:         |                                                                                                                                                    |
| **Optional**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **OptionalGetElement**        |     :x:      |        :x:         |                                                                                                                                                    |
| **OptionalHasElement**        |     :x:      |        :x:         |                                                                                                                                                    |
| **Or**                        |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **PRelu**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Pad**                       |     :x:      | :white_check_mark: | axes input not supported.                                                                                                                          |
| **Pow**                       |     :x:      | :white_check_mark: | No support for power with integer types.                                                                                                           |
| **QLinearConv**               |     :x:      |        :x:         |                                                                                                                                                    |
| **QLinearMatMul**             |     :x:      |        :x:         |                                                                                                                                                    |
| **QuantizeLinear**            |     :x:      | :white_check_mark: | Do not support per-axis and i7 quantization.                                                                                                       |
| **RNN**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **RandomNormal**              |     :x:      |        :x:         |                                                                                                                                                    |
| **RandomNormalLike**          |     :x:      |        :x:         |                                                                                                                                                    |
| **RandomUniform**             |     :x:      |        :x:         |                                                                                                                                                    |
| **RandomUniformLike**         |     :x:      |        :x:         |                                                                                                                                                    |
| **Range**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Reciprocal**                |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ReduceL1**                  |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceL2**                  |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceLogSum**              |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceLogSumExp**           |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceMax**                 |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceMean**                |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceMin**                 |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceProd**                |     :x:      | :white_check_mark: | do_not_keep_dim not supported.                                                                                                                     |
| **ReduceSum**                 |     :x:      | :white_check_mark: | Default axis and do_not_keep_dim not supported.                                                                                                    |
| **ReduceSumSquare**           |     :x:      | :white_check_mark: | Default axis and do_not_keep_dim not supported.                                                                                                    |
| **Relu**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Reshape**                   |     :x:      | :white_check_mark: | allowzero not supported.                                                                                                                           |
| **Resize**                    |     :x:      | :white_check_mark: | Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. |
| **ReverseSequence**           |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **RoiAlign**                  |     :x:      |        :x:         |                                                                                                                                                    |
| **Round**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **STFT**                      |     :x:      |        :x:         |                                                                                                                                                    |
| **SVMClassifier**             |     :x:      |        :x:         |                                                                                                                                                    |
| **SVMRegressor**              |     :x:      |        :x:         |                                                                                                                                                    |
| **Scaler**                    |     :x:      |        :x:         |                                                                                                                                                    |
| **Scan**                      |     :x:      | :white_check_mark: | Does not support dynamic shapes.                                                                                                                   |
| **Scatter**                   |     :x:      |        :x:         |                                                                                                                                                    |
| **ScatterElements**           |     :x:      | :white_check_mark: | Does not support duplicate indices.                                                                                                                |
| **ScatterND**                 |     :x:      | :white_check_mark: | Does not support scatternd add/multiply.                                                                                                           |
| **Selu**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **SequenceAt**                |     :x:      |        :x:         |                                                                                                                                                    |
| **SequenceConstruct**         |     :x:      |        :x:         |                                                                                                                                                    |
| **SequenceEmpty**             |     :x:      |        :x:         |                                                                                                                                                    |
| **SequenceErase**             |     :x:      |        :x:         |                                                                                                                                                    |
| **SequenceInsert**            |     :x:      | :white_check_mark: | Does not support unranked sequence element.                                                                                                        |
| **SequenceLength**            |     :x:      |        :x:         |                                                                                                                                                    |
| **SequenceMap**               |     :x:      |        :x:         |                                                                                                                                                    |
| **Shape**                     |     :x:      | :white_check_mark: | Does not support start and end attributes.                                                                                                         |
| **Shrink**                    |     :x:      |        :x:         |                                                                                                                                                    |
| **Sigmoid**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Sign**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Sin**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Sinh**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Size**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Slice**                     |     :x:      | :white_check_mark: | Axis must be a constant argument.                                                                                                                  |
| **Softmax**                   |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **SoftmaxCrossEntropyLoss**   |     :x:      |        :x:         |                                                                                                                                                    |
| **Softplus**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Softsign**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **SpaceToDepth**              |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Split**                     |     :x:      | :white_check_mark: | Does not support static and dynamic shape, zero size splits.                                                                                       |
| **SplitToSequence**           |     :x:      |        :x:         |                                                                                                                                                    |
| **Sqrt**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Squeeze**                   |     :x:      | :white_check_mark: | Does not support static and dynamic shape.                                                                                                         |
| **StringNormalizer**          |     :x:      |        :x:         |                                                                                                                                                    |
| **Sub**                       |     :x:      | :white_check_mark: | Does not support short integers.                                                                                                                   |
| **Sum**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Tan**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Tanh**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **TfIdfVectorizer**           |     :x:      |        :x:         |                                                                                                                                                    |
| **ThresholdedRelu**           |     :x:      |        :x:         |                                                                                                                                                    |
| **Tile**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **TopK**                      |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Transpose**                 |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **TreeEnsembleClassifier**    |     :x:      |        :x:         |                                                                                                                                                    |
| **TreeEnsembleRegressor**     |     :x:      |        :x:         |                                                                                                                                                    |
| **Trilu**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Unique**                    |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Unsqueeze**                 |     :x:      | :white_check_mark: | Does not support static and dynamic shape.                                                                                                         |
| **Upsample**                  |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Where**                     |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **Xor**                       |     :x:      | :white_check_mark: |                                                                                                                                                    |
| **ZipMap**                    |     :x:      |        :x:         |                                                                                                                                                    |
