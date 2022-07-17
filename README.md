# TIB-VA at SemEval-2022 Task 5: A Multimodal Architecture for the Detection and Classification of Misogynous Memes

![Model Architecture](src/architecture.png?raw=true "Model Architecture")


## Publication

[Paper](https://aclanthology.org/2022.semeval-1.105.pdf)

[Presentation recording](https://av.tib.eu/media/57745)

[SemEval-2022 Task 5](https://competitions.codalab.org/competitions/34175): MAMI - Multimedia Automatic Misogyny Identification, co-located with NAACL 2022


## Source code

Python (3.7) libraries: clip, torch, numpy, sklearn - "requirements.txt"

The model architecture code is in the file "train_multitask.py"


## Dataset

The dataset files are under "data". Images need to be downloaded and put under the parent folder "data" as "training_images" and "test_images". Download [link](https://drive.google.com/file/d/169qe9n4EbNlVbzFWNMjVX3N74Hh5Jcqr/view?usp=sharing).


## Cite
```bash
@inproceedings{hakimov-etal-2022-tib,
    title = "{TIB}-{VA} at {S}em{E}val-2022 Task 5: A Multimodal Architecture for the Detection and Classification of Misogynous Memes",
    author = "Hakimov, Sherzod  and
      Cheema, Gullal Singh  and
      Ewerth, Ralph",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.105",
    pages = "756--760"
}
```