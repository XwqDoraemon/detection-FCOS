checkpoint="training_dir/model_7.pth"
data_time=`date +%Y-%m-%d,%H:%M`
scrip=train.py
log_name=${data_time}${scrip}.log
python $scrip --log $log_name