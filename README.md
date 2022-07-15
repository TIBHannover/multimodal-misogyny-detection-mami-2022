# TIB-VA at SemEval-2022 Task 5: A Multimodal Architecture for the Detection and Classification of Misogynous Memes

![Model Architecture](src/architecture.png?raw=true "Model Architecture")


## Publication

[Arxiv version](https://arxiv.org/pdf/2204.06299.pdf)

[Presentation recording](https://av.tib.eu/media/57745)

[SemEval-2022 Task 5: MAMI - Multimedia Automatic Misogyny Identification, co-located with NAACL 2022](https://competitions.codalab.org/competitions/34175)


## Source code

Python libraries: clip, torch, numpy, sklearn

The model architecture code is in the file "train_multitask.py"


## Dataset

The dataset files are under "data". Images need to be downloaded and put under the parent folder "data" as "training_images" and "test_images". Download [link]().


## Cite
```bash
@inproceedings{tibva_mami2022,
  title={TIB-VA at SemEval-2022 Task 5: A Multimodal Architecture for the Detection and Classification of Misogynous Memes},
  author={Hakimov, Sherzod and Cheema, Gullal S. and Ewerth, Ralph},
  booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  year={2022}
}
```