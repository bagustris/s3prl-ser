<p align="center">
    <img src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/S3PRL-logo.png" width="900"/>
    <br>
    <br>
    <a href="./LICENSE.txt"><img alt="Apache License 2.0" src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/license.svg" /></a>
    <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="CC_BY_NC License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" /></a>
    <a href="https://github.com/s3prl/s3prl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/s3prl/s3prl/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
    <a href="#development-pattern-for-contributors"><img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
    <a href="https://github.com/s3prl/s3prl/issues"><img alt="Bitbucket open issues" src="https://img.shields.io/github/issues/s3prl/s3prl"></a>
</p>

# S3PRL-SER
S3PRL for Speech Emotion Recognition. See s3prl > downstream for supported speech emotion datasets.


## Environment compatibilities [![CI](https://github.com/s3prl/s3prl/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/s3prl/s3prl/actions/workflows/ci.yml)

We support the following environments. The test cases are ran with **[tox](./tox.ini)** locally and on **[github action](.github/workflows/ci.yml)**:

| Env | versions |
| --- | --- |
| os  | `ubuntu-18.04`, `ubuntu-20.04` |
| python | `3.7`, `3.8`, `3.9`, `3.10` |
| pytorch | `1.13.1` |

## What's New

* *Mar 2022*: Introduce [**SUPERB-SG**](https://arxiv.org/abs/2203.06849), see [Speech Translation](./s3prl/downstream/speech_translation), [Out-of-domain ASR](./s3prl/downstream/ctc/), [Voice Conversion](./s3prl/downstream/a2o-vc-vcc2020/), [Speech Separation](./s3prl/downstream/separation_stft/) and [Speech Enhancement](./s3prl/downstream/enhancement_stft/) for more info.
* *Nov 2021*: Introduce [**S3PRL-VC**](https://arxiv.org/abs/2110.06280), see [Any-to-one](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/a2o-vc-vcc2020) and [Any-to-any](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/a2a-vc-vctk) docs for more info.
* *Oct 2021*: Support [**DistilHuBERT**](https://arxiv.org/abs/2110.01900), see [docs](./s3prl/upstream/distiller/README.md) for more info
* *Sep 2021:* We host a *challenge* in [*AAAI workshop: The 2nd Self-supervised Learning for Audio and Speech Processing*](https://aaai-sas-2022.github.io/)! See [**SUPERB official site**](https://superbbenchmark.org/) for the challenge details and the [**SUPERB documentation**](./s3prl/downstream/docs/superb.md) in this toolkit!
* *Aug 2021:* We now have a tutorial that introduces our toolkit, you can **[watch it on Youtube](https://youtu.be/PkMFnS6cjAc)**!
* *July 2021:* We are now working on packaging s3prl and reorganizing the file structure in **v0.3**. Please consider using the stable **v0.2.0** for now. We will test and release **v0.3** before August.
* *June 2021:* Support [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark](https://arxiv.org/abs/2105.01051), submitted to Interspeech 2021. Use the tag **superb-interspeech2021** or **v0.2.0**.
* *June 2021:* Support extracting multiple hidden states from the SSL pretrained models
* *Jan 2021:* Readme updated with detailed instructions on how to use our latest version!
* *Dec 2020:* We are migrating to a newer version for a more general, flexible, and scalable code. See the introduction below for more information! The legacy version can be accessed the tag **v0.1.0**.

## Introduction and Usages

This is an open source toolkit called **s3prl**, which stands for **S**elf-**S**upervised **S**peech **P**re-training and **R**epresentation **L**earning.
Self-supervised speech pre-trained models are called **upstream** in this toolkit, and are utilized in various **downstream** tasks.

Unlike the original S3PRL, the S3PRL-SER has **a single usage** on Downstream:

### Downstream

- Utilize upstream models in lots of downstream tasks
- Benchmark upstream models with [**SUPERB Benchmark**](./s3prl/downstream/docs/superb.md)
- Document: [**downstream/README.md**](./s3prl/downstream/README.md)

Please refer to original S3PRL repository if you to experiment with **Pre-train** and **Upstream** usages.

Below is an **intuitive illustration** on how this toolkit may help you:
\
\
<img src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/S3PRL-interface.png" width="900"/>
\
\
Feel free to use or modify our toolkit in your research. Here is a [list of papers using our toolkit](#used-by). Any question, bug report or improvement suggestion is welcome through [opening up a new issue](https://github.com/s3prl/s3prl/issues). 

If you find this toolkit helpful to your research, please do consider citing [our papers](#citation), thanks!

## Installation

1. **Python** >= 3.8
2. Install **sox** on your OS
3. Install s3prl: [Read doc](https://s3prl.github.io/s3prl/tutorial/installation.html#) or `pip install -e ".[all]"`
4. (Optional) Some upstream models require special dependencies. If you encounter error with a specific upstream model, you can look into the `README.md` under each `upstream` folder. E.g., `upstream/pase/README.md`

## Development pattern for contributors

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/) of the [main S3PRL repository](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create a branch `new-awesome-feature`.
3. Contact us if you have any questions during development.
4. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/) through the Web interface of GitHub.
5. Please verify that your code is free of basic mistakes, we appreciate any contribution!

## Reference Repositories

* [Pytorch](https://github.com/pytorch/pytorch), Pytorch.
* [Audio](https://github.com/pytorch/audio), Pytorch.
* [Kaldi](https://github.com/kaldi-asr/kaldi), Kaldi-ASR.
* [Transformers](https://github.com/huggingface/transformers), Hugging Face.
* [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi), Mirco Ravanelli.
* [fairseq](https://github.com/pytorch/fairseq), Facebook AI Research.
* [CPC](https://github.com/facebookresearch/CPC_audio), Facebook AI Research.
* [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding), Yu-An Chung.
* [VQ-APC](https://github.com/s3prl/VQ-APC), Yu-An Chung.
* [NPC](https://github.com/Alexander-H-Liu/NPC), Alexander-H-Liu.
* [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu
* [Mockingjay](https://github.com/andi611/Mockingjay-Speech-Representation), Andy T. Liu.
* [ESPnet](https://github.com/espnet/espnet), Shinji Watanabe
* [speech-representations](https://github.com/awslabs/speech-representations), aws lab
* [PASE](https://github.com/santi-pdp/pase), Santiago Pascual and Mirco Ravanelli
* [LibriMix](https://github.com/JorisCos/LibriMix), Joris Cosentino and Manuel Pariente

## License

The majority of S3PRL Toolkit is licensed under the Apache License version 2.0, however all the files authored by Facebook, Inc. (which have explicit copyright statement on the top) are licensed under CC-BY-NC.



## Citation

If you find this toolkit useful, please consider citing following papers.

```
@article{Atmaja2022h,
  author = {Atmaja, Bagus Tris and Sasou, Akira},
  doi = {10.1109/ACCESS.2022.3225198},
  issn = {2169-3536},
  journal = {IEEE Access},
  pages = {124396--124407},
  title = {{Evaluating Self-Supervised Speech Representations for Speech Emotion Recognition}},
  url = {https://ieeexplore.ieee.org/document/9964237/},
  volume = {10},
  year = {2022}
}

@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
