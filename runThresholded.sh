# $1 is the port, $2 initial trial, $3 final trial, $4 number of agents: 

#In case number of agents is not informed, 1 is assumed
if test -z "$4" 
then
      $4=1
fi

for I in $(seq $2 1 $3)
do
python src/experiment.py -a1 C51ThresholdPolicy -a2 C51ThresholdPolicy -a3 C51ThresholdPolicy --number_trial $I --port $1 -n $4 -o $5
sleep 4
echo ********Completed $I
done
