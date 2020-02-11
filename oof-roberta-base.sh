# oof for 5 fold roberta-base, question_answer, seed 2020
python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Question_Answer" --max_len 512 --seed 901 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4

# oof for 5 fold roberta-base, question + answer, seed 123
python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Question" --max_len 512 --seed 123 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4
python oof-k-fold.py --model_type "bert" --model_name "roberta-base" --content "Answer" --max_len 512 --seed 123 --n_splits 5 --split "GroupKfold" --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4 --merge
