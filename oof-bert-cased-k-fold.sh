# oof for 10 fold bert-base-cased, question_answer, seed 1996
python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question_Answer" --seed 1996 --n_splits 10 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4


# oof for 5 fold bert-base-cased, question + answer, seed 1996
python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question" --seed 1996 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4
python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Answer" --seed 1996 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4 --merge

# oof for 5 fold bert-base-cased, question_answer, seed 726
# python oof-k-fold.py --model_type "bert" --model_name "bert-base-cased" --content "Question_Answer" --seed 726 --n_splits 5 --valid_batch_size 32 --split "GroupKfold" --loss "bce" --augment --swa --num_workers 4