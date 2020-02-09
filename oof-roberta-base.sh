
python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Question_Answer" --max_len 512 --seed 901 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4


# python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Question" --max_len 512 --seed 123 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4


# python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Answer" --max_len 512 --seed 123 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4 --merge
