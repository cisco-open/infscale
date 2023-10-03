# LLM-Inference
A Cisco Research project to optimize Large Language Model Inference systems. Internal project name: InfScale

## Code Structure
```
inference_pipeline.py -- the script that bears the PyTorch RPC-based implementation of ML pipeline.
optim_inference_pipeline.py -- the script that bears the optimized implementation of ML pipeline that combines both PyTorch RPC and low-level communication primitives.
partition_analysis.py -- the script that analyzes different partition strategies of ML models.
resnet50/ -- the directory that holds the scripts to run resnet50 inference and profiling as well as experiment results.
vgg16/ -- the directory that holds the scripts to run vgg16 inference and profiling as well as experiment results.
bert/ -- the directory that holds the scripts to run Bert inference and profiling as well as experiment results.
llama/ -- the directory that holds the scripts to run Llama inference and profiling results.
profiling/ -- the directory that holds the scripts to do profiling for ML models.
```

## PyTorch RPC-based implementation of LLM pipeline
A prototype implementation that uses [PyTorch RPC](https://pytorch.org/docs/stable/rpc.html) as the communication framework to construct a pipeline for ML models.
The *pipeline* is abstracted as a new ML model that wraps the original ML model and follows the original computation logic, but it uses a different computation process and aims to achieve better throughput in terms of samples processed per second.
The pipeline contains several *shards* each of which holds a *partition* of the original ML model.
All shards that bear the same partition are recognized as replicas of each other and they together form a *stage* of the pipeline.
The partitions held by different stages together should be able to be reconnected to reproduce the original ML model.
The communications across shards of different stages were executed through PyTorch RPC calls and RRefs.
The data dependencies across stages should follow the data dependencies across layers in the original ML model.
```
class CNNShardBase
```
The class that implements the abstraction of a shard of pipelined Convolutional Neural Networks.
```
class RR_CNNPipeline
```
The class that implements the abstraction of a pipeline for Convolutional Neural Networks.
```
class TransformerShardBase
```
The class that implements the abstraction of a shard of pipelined Transformer NN models.
```
class RR_TransformerPipeline
```
The class that implements the abstraction of a pipeline for Transformer NN models.

## Optimized implementation of LLM pipeline (Development Ongoing)
A prototype implementation that is renovated from the PyTorch RPC-based implementation with optimizations aiming to improve performance.
Although the instances of pipeline classes still use PyTorch RPC to configure and control shards managed by the pipeline in this implementation, the data communications between shards are using [PyTorch distributed p2p communication primitives](https://pytorch.org/docs/stable/distributed.html#point-to-point-communication).
```
class CNNShardBase
```
The class implements the abstraction of a shard of pipelined Convolutional Neural Networks.
```
class CNNPipelineCollector (To be corrected)
```
This class implements a collector that receives unordered results of mini-batches from shards of the last stage of the pipeline and reorders them to obtain the consistent result of the input batch data.
**This class is not completely correct yet! See the problem below:**
The objects of this class will start several threads to receive results from shards of the last pipeline stage. Since the object may be reused for different batches of input data, it is hard to stop those threads gracefully, i.e. terminate the threads with minimal user involvement and no restrictions on the operations of the Pipeline.
Right now, there seem to be two options:
- in the destruction function of CNNPipeline class, we explicitly call a method of CNNPipelineCollector to terminate all the threads
- every run of forwarding function of CNN Pipeline class will initialize a new instance of CNNPipelineCollector and then destroy it at the end.
```
class CNNPipeline
```
The class implements the abstraction of a pipeline for Convolutional Neural Networks.

