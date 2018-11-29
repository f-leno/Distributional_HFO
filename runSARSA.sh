# $1 is the port, $2 initial trial, $3 final trial: 
for I in $(seq $2 1 $3)
do
python src/experiment.py -a1 SARSA -a2 SARSA -a3 SARSA --number_trial $I --port $1
sleep 4
echo ********Completed $I
done
