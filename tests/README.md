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

| Op                            | Supported Opsets (inclusive) | Limitations                                                                                                                                        | ONNX-MLIR support  |
| ----------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------: |
| **Abs**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Acos**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Acosh**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Adagrad**                   | none                         |                                                                                                                                                    | :x: |
| **Adam**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Add**                       | none                         | No support for short integers.                                                                                                                     | :white_check_mark: |
| **And**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **ArgMax**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **ArgMin**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **ArrayFeatureExtractor**     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Asin**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Asinh**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Atan**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Atanh**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **AveragePool**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **BatchNormalization**        | none                         | Training not supported.                                                                                                                            | :white_check_mark: |
| **Bernoulli**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **Binarizer**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **BitShift**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **BitwiseAnd**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **BitwiseNot**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **BitwiseOr**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **BitwiseXor**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **BlackmanWindow**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **Cast**                      | none                         | Cast only between float and double types. Only ppc64le and MacOS platforms support float16.                                                        | :white_check_mark: |
| **CastLike**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **CastMap**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **CategoryMapper**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **Ceil**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Celu**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **CenterCropPad**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **Clip**                      | none                         | No support for short integers.                                                                                                                     | :white_check_mark: |
| **Col2Im**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **Compress**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Concat**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **ConcatFromSequence**        | none                         |                                                                                                                                                    | :white_check_mark: |
| **Constant**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **ConstantOfShape**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **Conv**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **ConvInteger**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **ConvTranspose**             | none                         | Unknown dimension in spatial dimensions (such as H and W) not supported.                                                                           | :white_check_mark: |
| **Cos**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Cosh**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **CumSum**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **DFT**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **DeformConv**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **DepthToSpace**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **DequantizeLinear**          | none                         | Only support for per-tensor or layer dequantization. No support for per-axis dequantization.                                                       | :white_check_mark: |
| **Det**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **DictVectorizer**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **Div**                       | none                         | No support for short integers.                                                                                                                     | :white_check_mark: |
| **Dropout**                   | none                         | Does not support masked and training.                                                                                                              | :white_check_mark: |
| **DynamicQuantizeLinear**     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Einsum**                    | none                         | Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32.               | :white_check_mark: |
| **Elu**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Equal**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Erf**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Exp**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Expand**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **EyeLike**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **FeatureVectorizer**         | none                         |                                                                                                                                                    | :white_check_mark: |
| **Flatten**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **Floor**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **GRU**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Gather**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **GatherElements**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **GatherND**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Gemm**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **GlobalAveragePool**         | none                         |                                                                                                                                                    | :white_check_mark: |
| **GlobalLpPool**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **GlobalMaxPool**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **Gradient**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Greater**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **GreaterOrEqual**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **GridSample**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **GroupNormalization**        | none                         |                                                                                                                                                    | :white_check_mark: |
| **HammingWindow**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **HannWindow**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **HardSigmoid**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **HardSwish**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **Hardmax**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **Identity**                  | none                         | Sequence identity not supported.                                                                                                                   | :white_check_mark: |
| **If**                        | none                         | Sequence and Optional outputs are not supported.                                                                                                   | :white_check_mark: |
| **Imputer**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **InstanceNormalization**     | none                         |                                                                                                                                                    | :white_check_mark: |
| **IsInf**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **IsNaN**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **LRN**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **LSTM**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **LabelEncoder**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **LayerNormalization**        | none                         |                                                                                                                                                    | :white_check_mark: |
| **LeakyRelu**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **Less**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **LessOrEqual**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **LinearClassifier**          | none                         |                                                                                                                                                    | :white_check_mark: |
| **LinearRegressor**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **Log**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **LogSoftmax**                | none                         | Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13.                                                                   | :white_check_mark: |
| **Loop**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **LpNormalization**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **LpPool**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **MatMul**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **MatMulInteger**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **Max**                       | none                         | No support for unsigned int. Only ppc64le and MacOS platforms support float16.                                                                     | :white_check_mark: |
| **MaxPool**                   | none                         | Does not support argmax and short ints. Support single output only.                                                                                | :white_check_mark: |
| **MaxRoiPool**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **MaxUnpool**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **Mean**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **MeanVarianceNormalization** | none                         |                                                                                                                                                    | :white_check_mark: |
| **MelWeightMatrix**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **Min**                       | none                         | Does not support unsigned numbers. Only ppc64le and MacOS platforms support float16.                                                               | :white_check_mark: |
| **Mish**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Mod**                       | none                         | Support float and double only. Only ppc64le and MacOS platforms support float16.                                                                   | :white_check_mark: |
| **Momentum**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Mul**                       | none                         | Does not support short integers.                                                                                                                   | :white_check_mark: |
| **Multinomial**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **Neg**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **NegativeLogLikelihoodLoss** | none                         |                                                                                                                                                    | :white_check_mark: |
| **NonMaxSuppression**         | none                         |                                                                                                                                                    | :white_check_mark: |
| **NonZero**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **Normalizer**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **Not**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **OneHot**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **OneHotEncoder**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **Optional**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **OptionalGetElement**        | none                         |                                                                                                                                                    | :white_check_mark: |
| **OptionalHasElement**        | none                         |                                                                                                                                                    | :white_check_mark: |
| **Or**                        | none                         |                                                                                                                                                    | :white_check_mark: |
| **PRelu**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Pad**                       | none                         | axes input not supported.                                                                                                                          | :white_check_mark: |
| **Pow**                       | none                         | No support for power with integer types.                                                                                                           | :white_check_mark: |
| **QLinearConv**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **QLinearMatMul**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **QuantizeLinear**            | none                         | Do not support per-axis and i8 quantization.                                                                                                       | :white_check_mark: |
| **RNN**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **RandomNormal**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **RandomNormalLike**          | none                         |                                                                                                                                                    | :white_check_mark: |
| **RandomUniform**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **RandomUniformLike**         | none                         |                                                                                                                                                    | :white_check_mark: |
| **Range**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Reciprocal**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **ReduceL1**                  | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceL2**                  | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceLogSum**              | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceLogSumExp**           | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceMax**                 | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceMean**                | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceMin**                 | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceProd**                | none                         | do_not_keep_dim not supported.                                                                                                                     | :white_check_mark: |
| **ReduceSum**                 | none                         | Default axis and do_not_keep_dim not supported.                                                                                                    | :white_check_mark: |
| **ReduceSumSquare**           | none                         | Default axis and do_not_keep_dim not supported.                                                                                                    | :white_check_mark: |
| **Relu**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Reshape**                   | none                         | allowzero not supported.                                                                                                                           | :white_check_mark: |
| **Resize**                    | none                         | Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. | :white_check_mark: |
| **ReverseSequence**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **RoiAlign**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Round**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **STFT**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **SVMClassifier**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **SVMRegressor**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **Scaler**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **Scan**                      | none                         | Does not support dynamic shapes.                                                                                                                   | :white_check_mark: |
| **Scatter**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **ScatterElements**           | none                         | Does not support duplicate indices.                                                                                                                | :white_check_mark: |
| **ScatterND**                 | none                         | Does not support scatternd add/multiply.                                                                                                           | :white_check_mark: |
| **Selu**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceAt**                | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceConstruct**         | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceEmpty**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceErase**             | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceInsert**            | none                         | Does not support unranked sequence element.                                                                                                        | :white_check_mark: |
| **SequenceLength**            | none                         |                                                                                                                                                    | :white_check_mark: |
| **SequenceMap**               | none                         |                                                                                                                                                    | :white_check_mark: |
| **Shape**                     | none                         | Does not support start and end attributes.                                                                                                         | :white_check_mark: |
| **Shrink**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sigmoid**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sign**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sin**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sinh**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Size**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Slice**                     | none                         | Axis must be a constant argument.                                                                                                                  | :white_check_mark: |
| **Softmax**                   | none                         |                                                                                                                                                    | :white_check_mark: |
| **SoftmaxCrossEntropyLoss**   | none                         |                                                                                                                                                    | :white_check_mark: |
| **Softplus**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Softsign**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **SpaceToDepth**              | none                         |                                                                                                                                                    | :white_check_mark: |
| **Split**                     | none                         | Does not support static and dynamic shape, zero size splits.                                                                                       | :white_check_mark: |
| **SplitToSequence**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sqrt**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Squeeze**                   | none                         | Does not support static and dynamic shape.                                                                                                         | :white_check_mark: |
| **StringNormalizer**          | none                         |                                                                                                                                                    | :white_check_mark: |
| **Sub**                       | none                         | Does not support short integers.                                                                                                                   | :white_check_mark: |
| **Sum**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Tan**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **Tanh**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **TfIdfVectorizer**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **ThresholdedRelu**           | none                         |                                                                                                                                                    | :white_check_mark: |
| **Tile**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **TopK**                      | none                         |                                                                                                                                                    | :white_check_mark: |
| **Transpose**                 | none                         |                                                                                                                                                    | :white_check_mark: |
| **TreeEnsembleClassifier**    | none                         |                                                                                                                                                    | :white_check_mark: |
| **TreeEnsembleRegressor**     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Trilu**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Unique**                    | none                         |                                                                                                                                                    | :white_check_mark: |
| **Unsqueeze**                 | none                         | Does not support static and dynamic shape.                                                                                                         | :white_check_mark: |
| **Upsample**                  | none                         |                                                                                                                                                    | :white_check_mark: |
| **Where**                     | none                         |                                                                                                                                                    | :white_check_mark: |
| **Xor**                       | none                         |                                                                                                                                                    | :white_check_mark: |
| **ZipMap**                    | none                         |                                                                                                                                                    | :white_check_mark: |
