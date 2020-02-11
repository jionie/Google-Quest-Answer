# Google-Quest-Answer
This repository contains codes for Google-Quest-Answer. 

## Structure for data
please arrange project folder as 
```plain
codes
└── all codes in this repo
```
```plain
input
└── google-quest-challenge
      ├── train.csv
      ├── test.csv
      ├── train_augment_final_with_clean.csv (in translation_data folder)
      ├── sample_submission.csv
      └── split
           └── ...
```
```plain
model
└── bert
└── xlnet 
└── ...
```

## Codes for Dataset
Please check codes for Dataset in "dataset" folder, you could run tests for (splitting train val sets, train_data_loader, val_data_loader, test_dataloader):
```bash
python3 dataset.py
```

## Codes for Model
Please check codes for Model in "model" folder, you could run tests for models, and you can use "check_model.ipynb" to check model architecture:
```bash
python3 model_bert.py
```

## Codes for Training
Please check codes for Training, you should change the path first then run:
```bash
./bert-uncased-k-fold.sh
```
```bash
./bert-cased-k-fold.sh
```
```bash
./xlnet-cased-k-fold.sh
```
```bash
./roberta-base-k-fold.sh
```
| single model           | hidden_layers | MIN_LR | config.hidden_dropout_prob |
| ---------------- |  ---- | ---- | ---- |
|bert-base-uncased, question_answer|[-1, -3, -5, -7, -9]|2e-6|0.1
|bert-base-uncased, question+answer|[-1, -3, -5, -7, -9]|2e-6|0
|bert-base-cased, question_answer|[-1, -3, -5, -7, -9]|2e-6|0.1
|bert-base-cased, question+answer|[-2, -4, -6, -8, -10]|2e-6|0.1
|xlnet-base-cased, question_answer|[-3, -4, -5, -6, -7]|1.5e-6|0
|xlnet-base-cased, question+answer|[-3, -4, -5, -6, -7]|2e-6|0
|roberta-base, question_answer|[-3, -4, -5, -6, -7]|1.5e-6|0
|roberta-base, question+answer|[-3, -4, -5, -6, -7]|2e-6|0

## Codes for SWA
Please check codes for simple SWA (not official codes), you should change the path first then run:
```bash
./swa-bert-base-uncased-k-fold.sh
```
```bash
./swa-bert-base-cased-k-fold.sh
```
```bash
./swa-xlnet-cased-k-fold.sh
```
```bash
./swa-roberta-base-k-fold.sh
```

## Codes for Getting oof
Please check codes for oof, you should change the path first then run:
```bash
./oof-bert-uncased-k-fold.sh
```
```bash
./oof-bert-cased-k-fold.sh
```
```bash
./oof-xlnet-cased-k-fold.sh
```
```bash
./oof-roberta-base-k-fold.sh
```

#### model performace (oof)
| single model           | oof |
| ---------------- |  ---- |
|bert-base-uncased, question_answer|0.403928|
|bert-base-uncased, question+answer|0.404822|
|bert-base-cased, question_answer|0.403596|
|bert-base-cased, question+answer|0.405100|
|xlnet-base-cased, question_answer|0.398455|
|xlnet-base-cased, question+answer|0.410154|
|roberta-base, question_answer|0.395185|
|roberta-base, question+answer|0.412353|

The oof files are in https://www.kaggle.com/jionie/qaallmodellogs

## Codes for inference
Please use "models-with-optimization-v5.ipynb" in "inference" folder, this is also available on https://www.kaggle.com/jionie/models-with-optimization-v5

## Codes for postprocessing
You can test postprocessing with all oof files and "test_postprocessing.py" in "postprocessing_optimization" folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)
