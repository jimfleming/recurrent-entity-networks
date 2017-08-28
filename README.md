# Recurrent Entity Networks

This repository contains an independent TensorFlow implementation of recurrent entity networks from [Tracking the World State with
Recurrent Entity Networks](https://arxiv.org/abs/1612.03969). This paper introduces the first method to solve all of the bAbI tasks using 10k training examples. The author's original Torch implementation is now available [here](https://github.com/facebook/MemNN/tree/master/EntNet-babi).

<img src="assets/diagram.png" alt="Diagram of recurrent entity network architecture" width="886" height="658">

## Results

Percent error for each task, comparing those in the paper to the implementation contained in this repository.

Task | EntNet (paper) | EntNet (repo)
--- | --- | ---
1: 1 supporting fact | 0 | 0
2: 2 supporting facts | 0.1 | 3.0
3: 3 supporting facts | 4.1 | ?
4: 2 argument relations | 0 | 0
5: 3 argument relations | 0.3 | ?
6: yes/no questions | 0.2 | 0
7: counting | 0 | 0
8: lists/sets | 0.5 | 0
9: simple negation | 0.1 | 0
10: indefinite knowledge | 0.6 | 0
11: basic coreference | 0.3 | 0
12: conjunction | 0 | 0
13: compound coreference | 1.3 | 0
14: time reasoning | 0 | 0
15: basic deduction | 0 | 0
16: basic induction | 0.2 | 0
17: positional reasoning | 0.5 | 1.7
18: size reasoning | 0.3 | 1.5
19: path finding | 2.3 | 0
20: agents motivation | 0 | 0
**Failed Tasks** | 0 | ?
**Mean Error** | 0.5 | ?

NOTE: Some of these tasks (16 and 19, in particular) required a change in learning rate schedule to reliably converge.

## Setup

1. Download the datasets by running [download_babi.sh](download_babi.sh) or from [The bAbI Project](https://research.facebook.com/research/babi/).
2. Run [prep_data.py](entity_networks/prep_data.py) which will convert the datasets into [TFRecords](https://www.tensorflow.org/programmers_guide/reading_data#standard_tensorflow_format).
3. Run `python -m entity_networks.main` to begin training on QA1.

## Major Dependencies

- TensorFlow v1.1.0

(For additional dependencies see [requirements.txt](requirements.txt))

## Thanks!

- Thanks to Mikael Henaff for providing details about their paper over Thanksgiving break. :)
- Thanks to Andy Zhang ([@zhangandyx](https://twitter.com/zhangandyx)) for helping me troubleshoot numerical instabilities.
- Thanks to Mike Young ([@mikalyoung](https://github.com/mikalyoung)) for providing results on some of the longer tasks.
