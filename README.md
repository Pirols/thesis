# Thesis

Evaluating multi-task learning impact on commonsense reasoning and WSD.

## Datasets

1. [Raganato et al.](https://aclanthology.org/E17-1010/)'s WSD evaluation framework which can be obtained as follows:

```bash
wget -P datasets/ http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip -d datasets/ datasets/WSD_Evaluation_Framework.zip
rm datasets/WSD_Evaluation_Framework.zip
```

2. [Lourie et al.](https://arxiv.org/abs/2103.13009)'s rainbow collection of datasets, which can be obtained as follows:

```bash
wget -P datasets/ https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/text-to-text/v1.0.rainbow.tar.gz
tar -xf datasets/v1.0.rainbow.tar.gz -C datasets
rm datasets/v1.0.rainbow.tar.gz
```

## Training

In order to train your own model you can execute, for instance, the following:

```bash
PYTHONPATH=$(pwd) python csd/train.py --data_path datasets --datasets_id socialiqa
```

To see all available parameters run the following instead:

```bash
PYTHONPATH=$(pwd) python esc/train.py --help
```

## Acknowledgments

This project is inspired by [ESC: Redesigning WSD with Extractive Sense Comprehension](https://aclanthology.org/2021.naacl-main.371/), available [here](https://github.com/SapienzaNLP/esc).
