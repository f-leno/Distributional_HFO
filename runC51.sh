# $1 is the port, $2 initial trial, $3 final trial: 
for I in $(seq $2 1 $3)
do
python src/experiment.py -a1 C51Agent -a2 C51Agent -a3 C51Agent --number_trial $I --port $1
done
