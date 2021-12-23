#python3 train.py --seed 100 --kfold 5 --model_name rf --n_cores 30 &
#python3 train.py --seed 100 --kfold 5 --model_name rf --n_cores 30 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/'

python3 train.py --seed 100 --kfold 10 --model_name rf --n_cores 10 &
python3 train.py --seed 100 --kfold 10 --model_name rf --n_cores 10 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name svm --n_cores 5 &
python3 train.py --seed 100 --kfold 10 --model_name svm --n_cores 5 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name logistic --n_cores 5 &
python3 train.py --seed 100 --kfold 10 --model_name logistic --n_cores 5 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name lgb --n_cores 10 &
python3 train.py --seed 100 --kfold 10 --model_name lgb --n_cores 10 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/'



python3 train.py --seed 100 --kfold 10 --model_name rf --n_cores 10 --score_type raw &
python3 train.py --seed 100 --kfold 10 --model_name rf --n_cores 10 --score_type raw --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name svm --score_type raw --n_cores 5 &
python3 train.py --seed 100 --kfold 10 --model_name svm --score_type raw --n_cores 5 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name logistic --score_type raw --n_cores 5 &
python3 train.py --seed 100 --kfold 10 --model_name logistic --score_type raw --n_cores 5 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/' &

python3 train.py --seed 100 --kfold 10 --model_name lgb --score_type raw --n_cores 10 &
python3 train.py --seed 100 --kfold 10 --model_name lgb --score_type raw --n_cores 10 --cutoff 0 --save_path '/tf/storage/result/ml_optimal_cutoff/'

#python3 train.py --seed 100 --model_name svm &
#python3 train.py --seed 100 --model_name rf &
#python3 train.py --seed 100 --model_name logistic &
#python3 train.py --seed 100 --model_name lgb &
#
#python3 train.py --seed 1 --model_name xgb &
#python3 train.py --seed 1 --model_name svm &
#python3 train.py --seed 1 --model_name rf &
#python3 train.py --seed 1 --model_name logistic &
#python3 train.py --seed 1 --model_name lgb
#
#sleep 30m
#
#python3 train.py --seed 2 --model_name xgb &
#python3 train.py --seed 2 --model_name svm &
#python3 train.py --seed 2 --model_name rf &
#python3 train.py --seed 2 --model_name logistic &
#python3 train.py --seed 2 --model_name lgb &
#
#
#python3 train.py --seed 3 --model_name xgb &
#python3 train.py --seed 3 --model_name svm &
#python3 train.py --seed 3 --model_name rf &
#python3 train.py --seed 3 --model_name logistic &
#python3 train.py --seed 3 --model_name lgb