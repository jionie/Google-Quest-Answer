python training-k-fold.py --model_type "xlnet" --content "Question_Answer" --model_name "xlnet-large-cased" --fold 0 --seed 666 --n_splits 5 --batch_size 2 --valid_batch_size 16 --accumulation_steps 4 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "xlnet" --content "Question_Answer" --model_name "xlnet-large-cased" --fold 1 --seed 666 --n_splits 5 --batch_size 2 --valid_batch_size 16 --accumulation_steps 4 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "xlnet" --content "Question_Answer" --model_name "xlnet-large-cased" --fold 2 --seed 666 --n_splits 5 --batch_size 2 --valid_batch_size 16 --accumulation_steps 4 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "xlnet" --content "Question_Answer" --model_name "xlnet-large-cased" --fold 3 --seed 666 --n_splits 5 --batch_size 2 --valid_batch_size 16 --accumulation_steps 4 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "xlnet" --content "Question_Answer" --model_name "xlnet-large-cased" --fold 4 --seed 666 --n_splits 5 --batch_size 2 --valid_batch_size 16 --accumulation_steps 4 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4



