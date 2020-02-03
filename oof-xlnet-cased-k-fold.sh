python oof-k-fold.py --model_type "xlnet" --model_name "xlnet-base-cased" --content "Question_Answer" --seed 1997 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4


# training using 768
# python oof-k-fold.py --model_type "xlnet" --model_name "xlnet-base-cased" --content "Question" --max_len 512 --seed 1997 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4



# python oof-k-fold.py --model_type "xlnet" --model_name "xlnet-base-cased" --content "Answer" --max_len 512 --seed 1997 --n_splits 5 --valid_batch_size 32 --loss "bce" --augment --swa --num_workers 4 --merge
