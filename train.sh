curpath=`realpath $0`
curdir=`dirname $curpath`
echo $curdir

# build vocab
python $curdir/code/corpus.py

# build PyG dataset
python $curdir/code/dataset.py --vocab_dir $curdir/model/vocab --mode train
python $curdir/code/dataset.py --vocab_dir $curdir/model/vocab --mode val

# train
python $curdir/code/train.py \
    --norm_size 96 \
    --vocab_dir $curdir/model/vocab \
    --embedding_dims 128 \
    --use_edge_attr \
    --seq_model lstm \
    --lstm_hidden_dims 128 \
    --lstm_layers 2 \
    --gnn_model gatedgcn-e \
    --gnn_hidden_dims 128 \
    --gnn_out_dims 128 \
    --gnn_layers 3 \
    --train_batch_size 84 \
    --batch_k 4 \
    --val_batch_size 84 \
    --train_num_each_epoch 400000 \
    --num_epochs 80 \
    --num_workers 8 \
    --milestones 40 60 \
    --miner_type multisimi \
    --loss_type multisimi \
    --learning_rate 0.001 \
    --early_stopping 40 \
    --precision 16 \
    --save_name lstm_gatedgcn-e
