# Recurrent Entity Networks

This repository contains an independent TensorFlow implementation of recurrent entity networks from [Tracking the World State with
Recurrent Entity Networks](https://openreview.net/forum?id=rJTKKKqeg). Thanks to Mikael Henaff for their help and answering questions about their paper over Thanksgiving break. :)

![Diagram of recurrent entity network](images/diagram.png)

## Setup

1. Download the datasets by running [download_datasets.sh](download_datasets.sh) or from [The bAbI Project](https://research.facebook.com/research/babi/).
2. Run [prep_datasets.py](prep_datasets.py) which will convert the datasets into [TFRecords](https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/index.html#standard_tensorflow_format).
3. Run `python -m entity_networks.main` to begin training on QA1.
4. Run `./run_all` to train on all tasks.

## Dependencies

- TensorFlow v0.11
