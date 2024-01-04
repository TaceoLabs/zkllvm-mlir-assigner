# zkML Test Suite

This document serves two purposes.

- It explains how to run tests for the zkML pipeline.
- It keeps track of the supported ONNX operations and the current pinned version
  of ONNX-MLIR.

## TLDR: How to run the Test Suite

In the home directory of this repository, after building the project, run the
following command for a quick check:

```
python tests/run.py --fast
```

If you have some time, run the same script without the `--fast` flag (in the
home directory, after building). Moreover, grab a coffee. This takes some time:

```
python tests/run.py
```

## Folder Structure

Inside the `/tests` folder (where you found this README.md) is a Python script
and two subfolders.

- `/Models` Inside this folder are test cases for pre-trained models. Currently,
  there are tests for two MNIST models, one of which is a CNN.
- `/Ops` Inside this folder, you can find specific test cases for supported ONNX
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
Therefore, we use ONNX-MLIR
[at this commit](https://github.com/onnx/onnx-mlir/tree/a04f518c1b0b8e4971d554c399bb54efc00b81db)
as it incorporates with this version of LLVM. This section keeps track of the
supported ONNX operations of the zkML frontend and its limitations. Keep in mind
that we inherit all limitations from ONNX-MLIR as well. To inspect their
limitations, click
[this link](https://github.com/onnx/onnx-mlir/blob/a04f518c1b0b8e4971d554c399bb54efc00b81db/docs/SupportedONNXOps-cpu.md).

For further information on the operations, see the
[ONNX documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

**Known Limitations:** Most of the operations only work on floating points at
the moment. We will add integer support for the applicable operations, but focus
on floating point arithmetic at the moment.

**Note:** This project is under active development. Expect drastic changes in
the future. We aim to support every ONNX operation supported by ONNX-MLIR (as
long as it is applicable for ZK).

| Op                            |    zkML Support    | ONNX-MLIR support  | Limitations                                        |
| ----------------------------- | :----------------: | :----------------: | -------------------------------------------------- |
| **Abs**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **Acos**                      |        :x:         | :white_check_mark: | Supported in assigner, no backend support          |
| **Acosh**                     |        :x:         | :white_check_mark: | Supported in assigner, no backend support          |
| **Adagrad**                   |        :x:         |        :x:         |                                                    |
| **Adam**                      |        :x:         |        :x:         |                                                    |
| **Add**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **And**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **ArgMax**                    | :white_check_mark: | :white_check_mark: | Apparently select_last_index always 1 in ONNX-MLIR |
| **ArgMin**                    | :white_check_mark: | :white_check_mark: | Apparently select_last_index always 1 in ONNX-MLIR |
| **ArrayFeatureExtractor**     |        :x:         |        :x:         |                                                    |
| **Asin**                      |        :x:         | :white_check_mark: |                                                    |
| **Asinh**                     |        :x:         | :white_check_mark: |                                                    |
| **Atan**                      |        :x:         | :white_check_mark: |                                                    |
| **Atanh**                     |        :x:         | :white_check_mark: |                                                    |
| **AveragePool**               | :white_check_mark: | :white_check_mark: |                                                    |
| **BatchNormalization**        |        :x:         | :white_check_mark: |                                                    |
| **Bernoulli**                 |        :x:         |        :x:         |                                                    |
| **Binarizer**                 |        :x:         |        :x:         |                                                    |
| **BitShift**                  |        :x:         |        :x:         |                                                    |
| **BitwiseAnd**                |        :x:         | :white_check_mark: |                                                    |
| **BitwiseNot**                |        :x:         |        :x:         |                                                    |
| **BitwiseOr**                 |        :x:         | :white_check_mark: |                                                    |
| **BitwiseXor**                |        :x:         | :white_check_mark: |                                                    |
| **BlackmanWindow**            |        :x:         |        :x:         |                                                    |
| **Cast**                      |        :x:         | :white_check_mark: |                                                    |
| **CastLike**                  |        :x:         |        :x:         |                                                    |
| **CastMap**                   |        :x:         |        :x:         |                                                    |
| **CategoryMapper**            |        :x:         |        :x:         |                                                    |
| **Ceil**                      | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Celu**                      |        :x:         |        :x:         |                                                    |
| **CenterCropPad**             |        :x:         |        :x:         |                                                    |
| **Clip**                      | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Col2Im**                    |        :x:         |        :x:         |                                                    |
| **Compress**                  |        :x:         | :white_check_mark: |                                                    |
| **Concat**                    |        :x:         | :white_check_mark: |                                                    |
| **ConcatFromSequence**        |        :x:         |        :x:         |                                                    |
| **Constant**                  |        :x:         | :white_check_mark: |                                                    |
| **ConstantOfShape**           |        :x:         | :white_check_mark: |                                                    |
| **Conv**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **ConvInteger**               |        :x:         |        :x:         |                                                    |
| **ConvTranspose**             |        :x:         | :white_check_mark: |                                                    |
| **Cos**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **Cosh**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **CumSum**                    |        :x:         | :white_check_mark: |                                                    |
| **DFT**                       |        :x:         |        :x:         |                                                    |
| **DeformConv**                |        :x:         |        :x:         |                                                    |
| **DepthToSpace**              |        :x:         | :white_check_mark: |                                                    |
| **DequantizeLinear**          |        :x:         | :white_check_mark: |                                                    |
| **Det**                       |        :x:         |        :x:         |                                                    |
| **DictVectorizer**            |        :x:         |        :x:         |                                                    |
| **Div**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Dropout**                   |        :x:         | :white_check_mark: |                                                    |
| **DynamicQuantizeLinear**     |        :x:         | :white_check_mark: |                                                    |
| **Einsum**                    |        :x:         | :white_check_mark: |                                                    |
| **Elu**                       |        :x:         | :white_check_mark: |                                                    |
| **Equal**                     | :white_check_mark: | :white_check_mark: |                                                    |
| **Erf**                       |        :x:         | :white_check_mark: |                                                    |
| **Exp**                       |        :x:         | :white_check_mark: |                                                    |
| **Expand**                    |        :x:         | :white_check_mark: |                                                    |
| **EyeLike**                   |        :x:         |        :x:         |                                                    |
| **FeatureVectorizer**         |        :x:         |        :x:         |                                                    |
| **Flatten**                   |        :x:         | :white_check_mark: |                                                    |
| **Floor**                     | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **GRU**                       |        :x:         | :white_check_mark: |                                                    |
| **Gather**                    |        :x:         | :white_check_mark: |                                                    |
| **GatherElements**            |        :x:         | :white_check_mark: |                                                    |
| **GatherND**                  |        :x:         | :white_check_mark: |                                                    |
| **Gemm**                      | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **GlobalAveragePool**         |        :x:         | :white_check_mark: |                                                    |
| **GlobalLpPool**              |        :x:         |        :x:         |                                                    |
| **GlobalMaxPool**             |        :x:         | :white_check_mark: |                                                    |
| **Gradient**                  |        :x:         |        :x:         |                                                    |
| **Greater**                   | :white_check_mark: | :white_check_mark: |                                                    |
| **GreaterOrEqual**            | :white_check_mark: | :white_check_mark: |                                                    |
| **GridSample**                |        :x:         |        :x:         |                                                    |
| **GroupNormalization**        |        :x:         |        :x:         |                                                    |
| **HammingWindow**             |        :x:         |        :x:         |                                                    |
| **HannWindow**                |        :x:         |        :x:         |                                                    |
| **HardSigmoid**               | :white_check_mark: | :white_check_mark: |                                                    |
| **HardSwish**                 |        :x:         |        :x:         |                                                    |
| **Hardmax**                   |        :x:         | :white_check_mark: |                                                    |
| **Identity**                  |        :x:         | :white_check_mark: |                                                    |
| **If**                        |        :x:         | :white_check_mark: |                                                    |
| **Imputer**                   |        :x:         |        :x:         |                                                    |
| **InstanceNormalization**     |        :x:         | :white_check_mark: |                                                    |
| **IsInf**                     |        :x:         | :white_check_mark: |                                                    |
| **IsNaN**                     |        :x:         | :white_check_mark: |                                                    |
| **LRN**                       |        :x:         | :white_check_mark: |                                                    |
| **LSTM**                      |        :x:         | :white_check_mark: |                                                    |
| **LabelEncoder**              |        :x:         |        :x:         |                                                    |
| **LayerNormalization**        |        :x:         | :white_check_mark: |                                                    |
| **LeakyRelu**                 | :white_check_mark: | :white_check_mark: |                                                    |
| **Less**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **LessOrEqual**               | :white_check_mark: | :white_check_mark: |                                                    |
| **LinearClassifier**          |        :x:         |        :x:         |                                                    |
| **LinearRegressor**           |        :x:         |        :x:         |                                                    |
| **Log**                       |        :x:         | :white_check_mark: |                                                    |
| **LogSoftmax**                | :white_check_mark: | :white_check_mark: |                                                    |
| **Loop**                      |        :x:         | :white_check_mark: |                                                    |
| **LpNormalization**           |        :x:         |        :x:         |                                                    |
| **LpPool**                    |        :x:         |        :x:         |                                                    |
| **MatMul**                    | :white_check_mark: | :white_check_mark: |                                                    |
| **MatMulInteger**             |        :x:         | :white_check_mark: |                                                    |
| **Max**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **MaxPool**                   | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **MaxRoiPool**                |        :x:         |        :x:         |                                                    |
| **MaxUnpool**                 |        :x:         |        :x:         |                                                    |
| **Mean**                      | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **MeanVarianceNormalization** |        :x:         |        :x:         |                                                    |
| **MelWeightMatrix**           |        :x:         |        :x:         |                                                    |
| **Min**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **Mish**                      |        :x:         |        :x:         |                                                    |
| **Mod**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Momentum**                  |        :x:         |        :x:         |                                                    |
| **Mul**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Multinomial**               |        :x:         |        :x:         |                                                    |
| **Neg**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **NegativeLogLikelihoodLoss** |        :x:         |        :x:         |                                                    |
| **NonMaxSuppression**         |        :x:         | :white_check_mark: |                                                    |
| **NonZero**                   |        :x:         | :white_check_mark: |                                                    |
| **Normalizer**                |        :x:         |        :x:         |                                                    |
| **Not**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **OneHot**                    |        :x:         | :white_check_mark: |                                                    |
| **OneHotEncoder**             |        :x:         |        :x:         |                                                    |
| **Optional**                  |        :x:         |        :x:         |                                                    |
| **OptionalGetElement**        |        :x:         |        :x:         |                                                    |
| **OptionalHasElement**        |        :x:         |        :x:         |                                                    |
| **Or**                        |        :x:         | :white_check_mark: |                                                    |
| **PRelu**                     | :white_check_mark: | :white_check_mark: |                                                    |
| **Pad**                       |        :x:         | :white_check_mark: |                                                    |
| **Pow**                       |        :x:         | :white_check_mark: |                                                    |
| **QLinearConv**               |        :x:         |        :x:         |                                                    |
| **QLinearMatMul**             |        :x:         |        :x:         |                                                    |
| **QuantizeLinear**            |        :x:         | :white_check_mark: |                                                    |
| **RNN**                       |        :x:         | :white_check_mark: |                                                    |
| **RandomNormal**              |        :x:         |        :x:         |                                                    |
| **RandomNormalLike**          |        :x:         |        :x:         |                                                    |
| **RandomUniform**             |        :x:         |        :x:         |                                                    |
| **RandomUniformLike**         |        :x:         |        :x:         |                                                    |
| **Range**                     |        :x:         | :white_check_mark: |                                                    |
| **Reciprocal**                | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **ReduceL1**                  | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceL2**                  | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceLogSum**              | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceLogSumExp**           | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceMax**                 | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceMean**                | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceMin**                 | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceProd**                | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceSum**                 | :white_check_mark: | :white_check_mark: |                                                    |
| **ReduceSumSquare**           | :white_check_mark: | :white_check_mark: |                                                    |
| **Relu**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **Reshape**                   |        :x:         | :white_check_mark: |                                                    |
| **Resize**                    |        :x:         | :white_check_mark: |                                                    |
| **ReverseSequence**           |        :x:         | :white_check_mark: |                                                    |
| **RoiAlign**                  |        :x:         |        :x:         |                                                    |
| **Round**                     | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **STFT**                      |        :x:         |        :x:         |                                                    |
| **SVMClassifier**             |        :x:         |        :x:         |                                                    |
| **SVMRegressor**              |        :x:         |        :x:         |                                                    |
| **Scaler**                    |        :x:         |        :x:         |                                                    |
| **Scan**                      |        :x:         | :white_check_mark: |                                                    |
| **Scatter**                   |        :x:         |        :x:         |                                                    |
| **ScatterElements**           |        :x:         | :white_check_mark: |                                                    |
| **ScatterND**                 |        :x:         | :white_check_mark: |                                                    |
| **Selu**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **SequenceAt**                |        :x:         |        :x:         |                                                    |
| **SequenceConstruct**         |        :x:         |        :x:         |                                                    |
| **SequenceEmpty**             |        :x:         |        :x:         |                                                    |
| **SequenceErase**             |        :x:         |        :x:         |                                                    |
| **SequenceInsert**            |        :x:         | :white_check_mark: |                                                    |
| **SequenceLength**            |        :x:         |        :x:         |                                                    |
| **SequenceMap**               |        :x:         |        :x:         |                                                    |
| **Shape**                     |        :x:         | :white_check_mark: |                                                    |
| **Shrink**                    |        :x:         |        :x:         |                                                    |
| **Sigmoid**                   | :white_check_mark: | :white_check_mark: |                                                    |
| **Sign**                      | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Sin**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **Sinh**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **Size**                      |        :x:         | :white_check_mark: |                                                    |
| **Slice**                     |        :x:         | :white_check_mark: |                                                    |
| **Softmax**                   | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **SoftmaxCrossEntropyLoss**   |        :x:         |        :x:         |                                                    |
| **Softplus**                  | :white_check_mark: | :white_check_mark: |                                                    |
| **Softsign**                  | :white_check_mark: | :white_check_mark: |                                                    |
| **SpaceToDepth**              |        :x:         | :white_check_mark: |                                                    |
| **Split**                     |        :x:         | :white_check_mark: |                                                    |
| **SplitToSequence**           |        :x:         |        :x:         |                                                    |
| **Sqrt**                      | :white_check_mark: | :white_check_mark: |                                                    |
| **Squeeze**                   |        :x:         | :white_check_mark: |                                                    |
| **StringNormalizer**          |        :x:         |        :x:         |                                                    |
| **Sub**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Sum**                       | :white_check_mark: | :white_check_mark: | No support for integers at the moment.             |
| **Tan**                       |        :x:         | :white_check_mark: |                                                    |
| **Tanh**                      |        :x:         | :white_check_mark: |                                                    |
| **TfIdfVectorizer**           |        :x:         |        :x:         |                                                    |
| **ThresholdedRelu**           |        :x:         |        :x:         |                                                    |
| **Tile**                      |        :x:         | :white_check_mark: |                                                    |
| **TopK**                      |        :x:         | :white_check_mark: |                                                    |
| **Transpose**                 |        :x:         | :white_check_mark: |                                                    |
| **TreeEnsembleClassifier**    |        :x:         |        :x:         |                                                    |
| **TreeEnsembleRegressor**     |        :x:         |        :x:         |                                                    |
| **Trilu**                     |        :x:         | :white_check_mark: |                                                    |
| **Unique**                    |        :x:         | :white_check_mark: |                                                    |
| **Unsqueeze**                 |        :x:         | :white_check_mark: |                                                    |
| **Upsample**                  |        :x:         | :white_check_mark: |                                                    |
| **Where**                     | :white_check_mark: | :white_check_mark: |                                                    |
| **Xor**                       | :white_check_mark: | :white_check_mark: |                                                    |
| **ZipMap**                    |        :x:         |        :x:         |                                                    |
