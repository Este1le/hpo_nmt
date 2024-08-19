# LCRankNet-v2

This directory contains scripts for a learning curve extrapolation model, `LCRankNet-v2`. It is a variation of `LCRankNet` ([Wistuba and Pedapati, 2020](https://arxiv.org/pdf/2006.03361)).

<img src="images/lcranknet_v2.png" alt="LCRankNet-v2" title="Architecture LCRankNet-v2." width="550" height="360">

`LCRankNet-v2` takes three inputs: partial learning curves, hyperparameter configurations, and task meta-information (including dataset ID, task type, source and target language, and base model). 

**Run training:**

```
python training.py --config lcranknet_configs.yaml
```

# Citation
```
@inproceedings{zhang2024best,
  	title={Best Practices of Successive Halving on Neural Machine Translation and Large Language Models},
  	author={Zhang, Xuan and Duh, Kevin},
  	booktitle={Proceedings of the 16th biennial conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)},
  	year={2024}
}

@inproceedings{wistuba2020learning,
  title={Learning to rank learning curves},
  author={Wistuba, Martin and Pedapati, Tejaswini},
  booktitle={International Conference on Machine Learning},
  pages={10303--10312},
  year={2020},
  organization={PMLR}
}
```