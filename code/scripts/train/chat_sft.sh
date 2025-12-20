nohup python sft/chat_sft.py >> exp_log/chatkbqa_sft_out.log  2>&1 &
echo $! > exp_log/save_pid.txt