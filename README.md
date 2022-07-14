# CrossNER

## Installation

The code is based on PyTorch 1.7.0 and Python 3.7.7. For training, a GPU is recommended to accelerate the training speed.

### Dependencies

The code is based on Python 3.7. Its dependencies are summarized in the file `requirements.txt`.

```
numpy==1.20.1
torch==1.7.0
tqdm==4.62.3
transformers==4.19.2
```

You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

### Code

The directory of code and its usage is:

```
./src
    ├── config.py // the hyper-parameters and the configuration
    ├── conll2002_metrics.py // the methods to calculate NER metrics
    ├── dataloader.py // load and parse the NER data
    ├── model.py // the main class for two-phase NER
    ├── trainer.py // the trainer for two-phase training and evaluation
    ├── utils.py // useful methods 
main.py // the entrance point for the whole process

```


