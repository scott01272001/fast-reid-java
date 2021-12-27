# fast-reid-java
fast-reid java api

## About this project
This project is the java API example for the [Fast-reid](https://github.com/JDAI-CV/fast-reid) which implements the state-of-the-art re-identification algorithms,
we use the [JavaCpp PyTorch preset](https://github.com/bytedeco/javacpp-presets/tree/master/pytorch) to bridge the [PyTorch C++ front-end API](https://pytorch.org/tutorials/advanced/cpp_frontend.html) with java to do the inference on the TorchScript format model.

## How to build
```
mvn clean package
```

## Use other model
The project uses the duke_sbs_S50 model by default, which can be download from [fast-reid model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md), 
then change the model to TorchScript format.

you can download the other model from the model zoo or do training by yourself.

## Project contain
- benchmark - Evaluate the execution time of the model on an image.
```
java -jar fast-reid-benchmark.jar
```

- fast-reid-java - fast-reid java API.
- fast-reid-test - Do test on fast-reid-java.
```
java -jar fast-reid-test.jar -m model_path -i input_image -d device(gpu or cpu) -index gpu_index
```

## Note
JvaCpp PyTorch preset uses cuda 11.2 and pytorch 1.10 by default, if the environment uses other versions of cuda, please download [LibTroch](https://pytorch.org/) that matches your cuda version and install the libtroch/lib/ to your system path or
include it in your system path, you can refer to [this](https://github.com/bytedeco/javacpp-presets/issues/1083).