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