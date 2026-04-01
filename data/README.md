# Data

This pipeline requires two publicly available datasets. Download them from Kaggle and place them in this directory.

## Dataset 1 - NeurIPS 2020 Education Challenge

15.8 million student interactions from the Eedi platform (UK secondary mathematics).

Download: https://www.kaggle.com/competitions/riiid-test-answer-prediction

Required file: `train_task_1_2.csv`

Place at: `data/train_task_1_2.csv`

## Dataset 2 - Eedi Mining Misconceptions in Mathematics

1,869 expert-labeled questions with verified misconception tags.

Download: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics

Required files: `train.csv` and `misconception_mapping.csv`

Place at: `data/train.csv` and `data/misconception_mapping.csv`

## Expected directory structure after download

```
data/
├── train_task_1_2.csv          (15.8M interactions, ~2GB)
├── train.csv                   (1,869 questions with text)
└── misconception_mapping.csv   (2,587 misconception labels)
```

Note: These files are excluded from the repository via `.gitignore` because of their size and licensing restrictions. You must download them directly from Kaggle.
