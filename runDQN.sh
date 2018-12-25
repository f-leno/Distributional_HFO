# $1 is the port, $2 initial trial, $3 final trial: 
for I in $(seq $2 1 $3)
do
python src/experiment.py -a1 DQNAgent -a2 DQNAgent -a3 DQNAgent --number_trial $I --port $1 -n $4 -o $5
sleep 4
echo ********Completed $I
done
