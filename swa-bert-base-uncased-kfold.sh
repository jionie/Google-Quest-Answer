# 10 fold bert-base-uncased, question_answer, seed 2020
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 1 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 2 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 3 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 4 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 5 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 6 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 7 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 8 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 9 --seed 2020 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain


# 5 fold bert-base-uncased, question + answer, seed 2020
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question" --max_len 512 --fold 0 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question" --max_len 512 --fold 1 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question" --max_len 512 --fold 2 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question" --max_len 512 --fold 3 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Question" --max_len 512 --fold 4 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain

python swa-k-fold.py --model_name "bert-base-uncased" --content "Answer" --max_len 512 --fold 0 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Answer" --max_len 512 --fold 1 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Answer" --max_len 512 --fold 2 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Answer" --max_len 512 --fold 3 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain
python swa-k-fold.py --model_name "bert-base-uncased" --content "Answer" --max_len 512 --fold 4 --seed 2020 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --split "GroupKfold" --lr 5e-7 --loss "bce" --augment --num_epoch 3 --num_workers 4 --load_pretrain


# new version bert-uncased for Ivan, 5 fold bert-base-uncased, question_answer, seed 1010
# python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 0 --seed 1010 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-6 --loss "bce" --split "GroupKfold" --augment --num_epoch 5 --num_workers 4 --load_pretrain
# python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 1 --seed 1010 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-6 --loss "bce" --split "GroupKfold" --augment --num_epoch 5 --num_workers 4 --load_pretrain
# python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 2 --seed 1010 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-6 --loss "bce" --split "GroupKfold" --augment --num_epoch 5 --num_workers 4 --load_pretrain
# python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 3 --seed 1010 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-6 --loss "bce" --split "GroupKfold" --augment --num_epoch 5 --num_workers 4 --load_pretrain
# python swa-k-fold.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 4 --seed 1010 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-6 --loss "bce" --split "GroupKfold" --augment --num_epoch 5 --num_workers 4 --load_pretrain