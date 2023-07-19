# Time Series Contrastive Learning

This project aims to curate state-of-the-art (SOTA) contrastive learning models for univariate and multi-variable time series learning. The goal is to provide a comprehensive list of currently implemented methods that can be used for time series analysis.

## What is Contrastive Learning and why do we need it for time series representation learning?
Contrastive learning is a type of unsupervised learning that aims to learn useful representations of data by contrasting similar and dissimilar pairs of examples. In the context of time series representation learning, contrastive learning is particularly useful because it allows us to learn representations without the need for explicit labels. This is important because labeling time series data can be difficult and time-consuming, especially for large datasets. Time series data, such as EMG signals, are particularly challenging to label because they often require domain expertise and manual review by experts. Contrastive learning can help overcome this challenge by learning representations that capture the underlying structure of the time series data, without the need for explicit labels.

## Currently Implemented Methods


Method	Description
| Name | Reference | Link |
| --- | --- | --- |
| CLOCS | Contrastive Learning on Convolutional Spectrograms, Kiyasseh et al. 2020 | [Link](https://arxiv.org/pdf/2005.13249) |
| Mixing-up | Mixing-up Data Augmentation for Time Series, Yang et al. 2022 | [Link](https://arxiv.org/pdf/2201.11739.pdf) |
| SimCLR | A Simple Framework for Contrastive Learning of Visual Representations, Chen et al. | [Link](https://arxiv.org/pdf/2002.05709) |
| TS-TCC | Time Series Transformation and Contrastive Coding for Time Series Classification Luo et al. | [Link](https://arxiv.org/pdf/2303.11911) |
| TS2Vec | Learning Distributed Representations of Time Series Yue, 2021 | [Link](https://arxiv.org/pdf/2106.10466) |
| TFC | Temporal Feature Clustering for Unsupervised Learning of Time Series Data | [Link](https://openreview.net/forum?id=OJ4mMfGKLN) |
| CLIP-motion | Joint Embedding learning between 1d motion data and activity labels. Shengzhi Li, unpublished. | N/A |


Each of these methods has been implemented and tested in this project. For more information on each method, please refer to the corresponding paper or documentation.
## Modifications to the original methods
### High-level goals
- Start from each model in its own directory, getting hands dirty with the code in terms of benchmarking, brining in modern practices (e.g. logging, distributed training, etc., large batch size, global contrastive loss, etc.) 
- Docuement and provide decision-tree over which model to use in univariate and multivariate settings.
### Progress to date
- July 19: Added my unpublished code named CLIP-motion. The code follows CLIP recipe to learn joint Embedding between 1d timeseries data of tri-axial IMU motion data and corresponding annotoations. In terms of data I created a mix of various human activity datasets totalling in 2.7 Million entries.  [Dataset Link](alexshengzhili/Accel2ActivityCrawl) 
- TFC training and evaluation scripts were re-written from scratch. The original code have numerous issues (for instance, I raised here: https://github.com/mims-harvard/TFC-pretraining/issues/23). 
- Contrastive learning benefits critically from large batch size. Huggingface Accelerate was integrated to support distributed training and gradient accumulation in TFC training code.
- TS2Vec has been modified to support end-to-end fine-tuning during fine-tuning stage. 
- TS2Vec can export sliding-window embeddings for fine-tuning. In our internal experiments, we found that fine-tuning on sliding-window embeddings can improve the performance of downstream tasks significantly comapred to only using globally pooled embeddings. 

## Usage

To use this project, simply clone the repository and run the desired method on your time series data. Each method has its own script and documentation, which can be found in the corresponding directory under `tsm/baselines/`.

## Contributing

If you would like to contribute to this project, please submit a pull request with your changes. We welcome contributions from the community and are always looking for ways to improve our methods and implementations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)

## Acknowledgments
We start off this project by using the code from [TFC] repository. 
