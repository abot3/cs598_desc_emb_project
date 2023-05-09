# CS598 Final Project

Group: 191: botelho3@illinois.edu, ~~tconst4@illinois.edu~~


**TODO: Add PDF to repo**


## Project Documentation
- Report 
    - [Google Docs Link](https://docs.google.com/document/d/1bLIQ81IsHfF2ZSuUE5uEzSp8a0GFN5Y4CCjlqLw2K_w/edit?usp=sharing)
- Presentation
    - [Google Slides Link](https://docs.google.com/presentation/d/1KU8uNUgng8C4sKGzMCqePgAUSKTsTT_cqCNMdvTBuas/edit?usp=sharing)
- Video
    - [Youtube Link](https://youtu.be/t75XAS6lhkw)
- Descriptive Notebook
    - See `notebook_with_plots.ipynb`
- Pretrained Models
    - [Google Drive Link](https://drive.google.com/drive/folders/1y2tbYFiqKp4o-ZhxbwwP0C8oRJucWfdc?usp=sharing)

## Name
Replicating Unifying Electronic Health Record Systems via Text Embedding

Original Paper:
```
@misc{
    hur2022unifying,
    title={Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding}, 
    author={Kyunghoon Hur and Jiyoung Lee and Jungwoo Oh and Wesley Price and Young-Hak Kim and Edward Choi},
    year={2022},
    eprint={2108.03625},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Original author's [repo on Github](https://github.com/hoon9405/DescEmb).

## Description

This repo attempts to replicate the paper `Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding`. It processes the eICU and MIMIC-III databases into an embedding using either a Text Encoder model or a RNN trained on sequences of medical events. The Text Encoder model is swappable, in practice we use BERT and BERT-Tiny. A Bidirectional-RNN using GRUs is then trained to make medical outcome predictions (e.g. Mortality) using the Medical Code or Medical Text embeddings from the Encoder layer as input. The .ipynb contains code to train, evaluate and plot performance of both models.

## Installation
Spin up a Conda environment and make sure to install the packages listed in `install.sh`. The major dependencies are PyTorch, PyHealth, BERT, Transformers and the standard Numpy/ScikitLearn/Pandas/Matplotlig/Jupyter stack.

If using conda create an environment and install dependencies:

```
conda create --name <env> --file requirements.txt
```
or 

```
cd <repo clone dir>
conda create --name <env_name>
conda activate <env_name>
./install.sh
```

Note: this project was tested on Linux. If on windows install using WSL2.

This project also requires the MIMIC-III and eICU datasets. Place these in a `home` directory relative path e.g. `~/ehr_data/*` and provide the path to the `.ipynb` in global vars `MIMIC_DATA_DIR_` and `EICU_DATA_DIR_`. If you have a GPU with > 4GB RAM you can enable/disable GPU eval and training via `BERT_USE_GPU_` and `USE_GPU_`. Dev mode can be enabled using the `DEV_` env variable, this loads a small subset of either dataset to test new code.

**MIMIC-III Dataset**

This project uses MIMIC-III v1.4 dataset. Available from [Physionet](https://physionet.org/content/mimiciii/1.4/). This dataset is close-source and requires a free training course and license agreement before it can be accessed. Details available at the MIT MIMIC project [homepage](https://mimic.mit.edu/docs/).

**eICU Dataset**

This project uses the eICU v2.0 database. Available from [Physionet](https://physionet.org/content/eicu-crd/2.0/). This dataset is close-source and requires a free training course and license agreement before it can be accessed. Details available at the MIT MIMIC project [homepage](https://eicu-crd.mit.edu/about/eicu/).

&nbsp;

## Source Layout 

### Trainlib

`trainer.py`

Contains Trainer class that accepts a dict of args controlling Model training parameters. E.g. `args.embed_model_type` and `args.predict_model_type` control the type of embedding and prediction model instantiated. Can be `desc_emb_ft`, `desc_emb`, or `code_emb`.  `args.task` and `args.db_name` control the prediction task requested e.g Mortality, LOS, or Readmission. The DB name controls which EHR datset is used for train or eval. `args.eval_only` controls wether training or eval is requested. Set to False to train + eval model. Set to true to evaluate model on test dataset. `args.collate_fn` and `args.no_use_cached_dataset` allow you to pass a Torch.Dataset and a torch `collate_fn` to the Trainer in place of loading a preprocessed dataset already saved on disk. The latter is much faster but less flexible. Common training hyperparameters can be set via `args.n_epochs`, `args.learning_rate`. `args.is_dev` can be used to load different cached dataset files from disk for dev mode and full mode, i.e. a different pathname is used to load data.

&nbsp;

### Tasks

`code_emb_funcs.py`, `desc_emb_funcs.py`, `eicu_funcs.py`

The above  three files contain [pyHealth Task functions](https://pyhealth.readthedocs.io/en/latest/api/tasks.html) that process the relevant dataset (eICU, MIMIC-III) into a sequence of [PyHealth Visits](https://pyhealth.readthedocs.io/en/latest/api/data/pyhealth.data.Visit.html).

In the case of CodeEmb Visits are sequences of ICD9/ICD10 codes that are transformed into a Tensor representing a sequence of multi-hot vectors, 1 per visit, per patient `(Batch, #Visits, #Events, MultiHotEmbDim)`. The CodeEmb collate function sums the multi-hot vectors along the #Events dimension resulting in a Tensor with dim `(Batch, #Visits, MultiHotEmbDim)`.

In the case of DescEmb Visits are sequences of natural language event descriptions (e.g. 'Aspirin 81mg Tab' or 'Colonoscopy', or 'Calculated Total CO2'). The sequences are embedded using the Text Encoder (BERT, RNN, BERT-Tiny) to a vector of length `embedding_dim` creating a Tensor with dimensions `(Batch, #MedicalEvents/Patient, EmbeddingDim)`.  Embedding dim is 768 for BERT and 128 for BERT-Tiny.

The embeddings generated by CodeEmb or DescEmb are then fed to a Bi-directional RNN prediction layer. This is the `predict_model_type`. The input is shape `(Batch, #MedicalEvents/Patient, EmbeddingDim)` for DescEmb. The input is shape `(Batch, #Visits, MultiHotEmbDim)` for CodeEmb.


`collate_funcs.py`

Contains the different collate functions for each of the 3 embedding model types \[`code_emb`, `desc_emb_ft`, `desc_emb`\].


`dataset_transforms.py`

Contains `class TextEmbedDataset` which extends [PyHealth's SampleDataset](https://pyhealth.readthedocs.io/en/latest/api/datasets.html) with a caching & transform function. The transform can be used to apply the `BertTextEmbedTransform` or `BertFineTuneTransform` which computes text embeddings for each sample. This can be used to preprocess the samples in your dataset and cache them to disk to avoid doing it during each training epoch.

&nbsp;

### Models

`ehr_model.py`

Extends `nn.Module` and accepts the top level `Trainer` args struct. Uses the 2 parameters \[`args.embed_model_type`, `args.predict_model_type`\] to load the models needed for CodeEmb, DescEmb or DescEmbFineTune prediction. This class contains no nn layers, just 2 models. The forwards pass invokes the embedding model, then passes the embedding model output to the prediction model.

```
def forward(self, **kwargs):
    x, rev_x = self.embed_model(**kwargs)  # (B, S, E)
    kwargs['x'] = x
    kwargs['rev_x'] = rev_x
    net_output = self.pred_model(**kwargs)
```

`code_emb.py` 

Implements the CodeEmb models: `class CembEmbed` and `class CembRNN`.

`desc_emb.py` 

Implements the DescEmb models: `class DembEmbed` and `class DembRNN`. Note: DembEmbed is a passthrough (Identity) layer because when BERT is not updated the BERT embeddings can be computed once during preprocessing in `TextEmbedDataset + BertTextEmbedTransform`.

`desc_emb_fine_tune.py` 

Implements the DescEmb models: `class DembFtEmbed` and `class DembFtRNN`. Note: DembFtEmbed invokes BERT to compute text embeddings, and BERT weights are updated during training. This is __slow__ and GPU is __required__ to train this model in a reasonable amount of time.

&nbsp;

### Datsets

`dataset.py`

Implements `class DatasetCacher` which accepts a `torch.Dataset`, a batch size, and a dataset length and writes the dataset to file using `pickle`. The filenames are hashes derived from the input arguments preventing collisions. This is used to store preprocessed `torch.Datasets` to disk.

Implements `class StructuredDataset`, a wrapper around `torch.Dataset` which uses a `metadata` dictionary to combine keywords and a sample tuple `zip(['x', 'rev_x', 'label'], samples_tuple)` into a dictionary. This dictionary is passed to `class EHRModel.forward(**kwargs)` as the kwargs argument during training/eval. The metadata dictionary is part of the preprocessed dataset stored on-disk in `pickle` format. This saves compute when loading samples each epoch.

&nbsp;

## Usage

Open the `.ipynb` file. Run the cells under the `Preprocessing` header to load, process, and save a dataset to disk before traning. Load one of the datasets using `Load MIMIC III Data` or `Load eICU Data` cells. Next run the `Dataloaders and Collate` cell corresponding to one of the three tasks you wish to perform ( `desc_emb_ft`, `desc_emb`, or `code_emb`) on the loaded dataset. Then cache the preprocessed dataset to disk using `Dataset Caching` cells.

To train a model on the cached dataset use one of the `Condensed Traning using Trainer` cells corresponding to the EHR prediction task you are interested in (e.g. Morality, Readimission, Length-of-Stay). This will load the cached data, create a `Trainer` object containing a `class EHRModel`, and invoke its `train()` function on the model. Eval on the test set is performed by `Trainer` and results are returned and plotted. If __NOT__ interested in training, a pretrained model can be evaluated by setting `args.eval_only = True`. Note: you have to train at least once in order to load a pre-trained model, they are not included in the repo.

## Results

<br>

**Claim 1**

Outcome: Reproducible

|               | **Mortablity<br>CodeEmb**     | **Mortality<br>DescEmb**     | **Readmission<br> CodeEmb**   | **Readmission DescEmb**      |
|---------------|-------------------------------|------------------------------|-------------------------------|------------------------------|
| **AUPRC**     | 0.61<br>(4681/28745 = 16% TP) | 0.7<br>(1519/11835 = 12% TP) | 0.84<br>(6529/26897 = 24% TP) | 0.86<br>(3363/9991 = 33% TP) |
| **AUROC**     | 0.88                          | 0.92                         | 0.89                          | 0.91                         |
| **Accuracy**  | 0.89                          | 0.91                         | 0.94                          | 0.94                         |
| **Precision** | 0.66                          | 0.76                         | 0.99                          | 0.86                         |
| **Recall**    | 0.47                          | 0.37                         | 0.72                          | 0.78                         |

<br>
<br>

**Claim 2**

Outcome: Not Reproducible

| **null**     | **DescEmb MIMIC-III**    | **DescEmb eICU**           |
|--------------|--------------------------|----------------------------|
| **AUPRC**    | 0.7 (1519/11835= 12% TP) | 0.2 (1266/15117 = 8.3% TP) |
| **AUROC**    | 0.92                     | 0.75                       |
| **Accuracy** | 0.88                     | 0.92                       |


<br>
<br>

**Claim 3**

Outcome: Partially Reproducible

|               | **Mortality CodeEmb**         | **Mortality DescEmb**        | **Mortality DescEmbFt**       |
|---------------|-------------------------------|------------------------------|-------------------------------|
| **AUPRC**     | 0.61<br>(4681/28745 = 16% TP) | 0.7<br>(1519/11835 = 12% TP) | 0.45<br>(1519/11835 = 12% TP) |
| **AUROC**     | 0.88                          | 0.92                         | 0.84                          |
| **Accuracy**  | 0.89                          | 0.91                         | 0.87                          |
| **Precision** | 0.66                          | 0.76                         | 0.54                          |
| **Recall**    | 0.47                          | 0.37                         | 0.19                          |



## Citations

```
@misc{hur2022unifying,
      title={Unifying Heterogeneous Electronic Health Records Systems via Text-Based Code Embedding}, 
      author={Kyunghoon Hur and Jiyoung Lee and Jungwoo Oh and Wesley Price and Young-Hak Kim and Edward Choi},
      year={2022},
      eprint={2108.03625},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{zhao2021pyhealth,
      title={PyHealth: A Python Library for Health Predictive Models}, 
      author={Yue Zhao and Zhi Qiao and Cao Xiao and Lucas Glass and Jimeng Sun},
      year={2021},
      eprint={2101.04209},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1423",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
}
```

```
Johnson, A., Pollard, T., & Mark, R. (2016).
MIMIC-III Clinical Database (version 1.4).
PhysioNet. https://doi.org/10.13026/C2XW26.
```

```
Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible critical care database.
Sci Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
```

## License
Copyright 2023 botelho3@illinois.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Project status

Frozen.