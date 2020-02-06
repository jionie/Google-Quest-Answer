# python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question_Answer" --seed 1996 --n_splits 10 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4



# python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question" --seed 1996 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4



# python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Answer" --seed 1996 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4 --merge


python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question_Answer" --seed 726 --n_splits 5 --valid_batch_size 32 --split "MultiStratifiedKfold" --loss "bce" --augment --swa --num_workers 4