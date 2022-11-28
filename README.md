# TIB-VA at SemEval-2022 Task 5: A Multimodal Architecture for the Detection and Classification of Misogynous Memes

![Model Architecture](src/architecture.png?raw=true "Model Architecture")


## Publication

[Paper](https://aclanthology.org/2022.semeval-1.105.pdf)

[Presentation recording](https://av.tib.eu/media/57745)

[SemEval-2022 Task 5](https://competitions.codalab.org/competitions/34175): MAMI - Multimedia Automatic Misogyny Identification, co-located with NAACL 2022


## Source code

Python (3.7) libraries: clip, torch, numpy, sklearn - "requirements.txt"

The model architecture code is in the file "train_multitask.py"

## Model Output from MAMI 2022

We provided the model outputs for Task A & B under the directory "mami_submissions". It includes the best submissions for each task ("answer.txt").


## Dataset

The dataset files are under "data". Images need to be downloaded and put under the parent folder "data" as "training_images" and "test_images". Download [link](https://drive.google.com/file/d/169qe9n4EbNlVbzFWNMjVX3N74Hh5Jcqr/view?usp=sharing).


## Cite
Please cite if you find the resource useful:
```bash
@inproceedings{DBLP:conf/semeval/HakimovCE22,
  author    = {Sherzod Hakimov and
               Gullal Singh Cheema and
               Ralph Ewerth},
  editor    = {Guy Emerson and
               Natalie Schluter and
               Gabriel Stanovsky and
               Ritesh Kumar and
               Alexis Palmer and
               Nathan Schneider and
               Siddharth Singh and
               Shyam Ratan},
  title     = {{TIB-VA} at SemEval-2022 Task 5: {A} Multimodal Architecture for the
               Detection and Classification of Misogynous Memes},
  booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation,
               SemEval@NAACL 2022, Seattle, Washington, United States, July 14-15,
               2022},
  pages     = {756--760},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://doi.org/10.18653/v1/2022.semeval-1.105},
  doi       = {10.18653/v1/2022.semeval-1.105},
  timestamp = {Mon, 01 Aug 2022 17:09:21 +0200},
  biburl    = {https://dblp.org/rec/conf/semeval/HakimovCE22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```