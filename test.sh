curpath=`realpath $0`
curdir=`dirname $curpath`
echo $curdir

# test
python $curdir/code/predict.py $curdir/model/lstm_gatedgcn-e/version_0