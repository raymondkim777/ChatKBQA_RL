python data_preprocess/chat.py
python data_preprocess/chat_sft_generate.py --split train >> exp_log/chatkbq_data.log 2>&1
echo $! > exp_log/save_pid.txt
python data_preprocess/chat_sft_generate.py --split test >> exp_log/chatkbq_data.log 2>&1
echo $! > exp_log/save_pid.txt
python data_preprocess/chat_sft.py --split train
python data_preprocess/chat_sft.py --split test