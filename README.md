# Google-Quest-Answer
This repository contains codes for Google-Quest-Answer. 

## Structure for data
please arrange input folder as 
```plain
input
└── google-quest-challenge
      ├── train.csv
      ├── test.csv
      ├── sample_submission.csv
      └── split
           └── ...
```

## Codes for Dataset
Please check codes for Dataset in "dataset" folder, you could run tests for (splitting train val sets, train_data_loader, val_data_loader, test_dataloader):
```bash
python3 dataset.py
```

## Codes for Training
Please check codes for Training, you should change the path first then run:
```bash
./bert-uncased-k-fold.sh
```

## Codes for Getting oof
Please check codes for oof, you should change the path first then run:
```bash
./oof-bert-uncased-k-fold.sh
```
