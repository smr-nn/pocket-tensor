# pocket-tensor

pocket-tensor is an [arquolo's](https://github.com/arquolo) [Kerasify](https://github.com/moof2k/kerasify) fork designed for running trained Keras models from a C++ application on embedded devices.

## Design goals

* Compatibility with sequential networks generated by Keras 2.x using Tensorflow backend.
* Multithread CPU support (no GPU support).
* Low RAM usage.
* Easy to build and run (no external dependencies).
* Fast build times.

## Improvements over Kerasify

* Thanks to the awesome [libsimdpp library](https://github.com/p12tic/libsimdpp), tensor operations have been rewritten using SIMD instructions to improve prediction performance.
* Predictions run across multiple CPU cores.
* Memory (re)usage has been improved in order to reduce memory allocations.
* Apart from `float`, `double` precision tensors are supported (see `pt_tweakme.h` file).
* Tensor dimensions are rigorously validated on each layer to avoid bad models usage.
* Besides GCC and Clang, Visual Studio compiler is properly supported.

## Hardware requirements

Since there's no GPU support, by default pocket-tensor requires the following CPU SIMD instruction sets:

* ARM: NEON with floating point support.
* x86: AVX.

Required SIMD instruction sets are specified in the `pt_tweakme.h` file, so they can be modified with ease.

## Software requirements

Since a copy of libsimdpp comes bundled with this library, there's no external dependencies required, so the only software requirements are a C++11-compatible compiler and CMake >= 3.4.  

pocket-tensor has been tested with these compilers: 

* GCC 4.9.
* MSVC 2017.
* Whatever Clang comes with Apple LLVM 9.1.0.
* Whatever Clang comes with Android Studio 3.1.3 (see Android section).

## How to build

A CMakeLists.txt is provided with this library, so in order to use it you only need to include this file in your CMake project.  

To build and run the unit tests, you need to generate them first:

```
python make_tests.py
mkdir tests_build
cd tests_build
cmake -DPT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
make
./tests/pocket-tensor-tests
```

## Usage

1) Use Keras to build (`model.compile(...)`) and train (`model.fit(...)`) your model as usual.

2) Now convert it to the Kerasify file format with `kerasify.export_model(model, 'example.model')`.

3) Finally load it in C++ (`pt::create("example.model")`) and use `model->predict(...)` to perform a prediction with your data.

The following example shows the full workflow:

```python
# make_model.py:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from kerasify import export_model

test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')

model = Sequential()
model.add(Dense(1, input_dim=10))

model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit(test_x, test_y, epochs=1)

print model.predict(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))

export_model(model, 'example.model')
```

```cpp
// main.cpp:

#include <iostream>
#include "pt_model.h"
#include "pt_tensor.h"

int main()
{
    // Initialize model:
    auto model = pt::create("example.model");
    // REQUIRE(model);

    // Create input tensor:
    pt::Tensor in(10);
    in.setData({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    // Run prediction:
    pt::Tensor out;
    bool success = model->predict(std::move(in), out);
    // REQUIRE(success);
	
    // Print output:
    std::cout << out << std::endl;
    return 0;
}
```

## Supported layer types

The most common layer types used in image recognition and sequences prediction are supported, making many popular model architectures possible:

* Convolutions: `Conv1D`, `Conv2D`, `LocallyConnected1D`.
* Sequences related: `LSTM`, `Embedding`.
* Activations: `Linear`, `ReLU`, `ELU`, `SeLU`, `LeakyReLU`, `Softplus`, `Softsign`, `Tanh`, `Sigmoid`, `HardSigmoid`, `Softmax`.
* Other: `Dense`, `Flatten`, `MaxPooling2D`, `BatchNormalization`, `ELU`.

## Performance

A benchmark application is included with this library. To build and run it:

```
mkdir benchmark_build
cd benchmark_build
cmake -DPT_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release ..
make
./benchmark/pocket-tensor-benchmark
```

The prediction time of the following models has been measured on a PC with a Intel Core i7-6500U CPU @ 2.50GHz and on a Raspberry Pi 3:

### Mnist

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))
```

| Library            | PC elapsed time (μs) | RPi3 elapsed time (μs) |
| ------------------ | -------------------: | ---------------------: |
| Keras              |                 1470 |                  23363 |
| arquolo's Kerasify |                 3502 |                  64238 |
| frugally-deep      |                 1402 |                  29298 |
| pocket-tensor      |                 1049 |                  27329 |

### Imdb

```python
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```

| Library            | PC elapsed time (μs) | RPi3 elapsed time (μs) |
| ------------------ | -------------------: | ---------------------: |
| Keras              |                10160 |                  89344 |
| arquolo's Kerasify |                 5378 |                  79060 |
| frugally-deep      |        Not supported |          Not supported |
| pocket-tensor      |                 3314 |                  67115 |

## Android

pocket-tensor supports Android apps (armeabi-v7a ABI only).

To add pocket-tensor to an Android project with C++ support, you must:

1) Enable ARM NEON instructions on the build.gradle project file (https://developer.android.com/ndk/guides/cmake):

```
android {
    ...
    defaultConfig {
        ...
        externalNativeBuild {
            cmake {
                arguments "-DANDROID_ARM_NEON=TRUE"
            }
        }
    }
}
```

2) Disable all ABIs except armeabi-v7a on the build.gradle project file (https://developer.android.com/studio/build/configure-apk-splits):

```
android {
    ...
    splits {
        abi {
            enable true
            reset()
            include "armeabi-v7a"
        }
    }
}
```

3) Include pocket-tensor on the CMakeLists.txt file of your native library:

```
add_subdirectory(/path/to/pocket-tensor pocket-tensor)
target_link_libraries(native-lib pocket-tensor)
```

