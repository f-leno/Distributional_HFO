pkill -f  sh\ ru* 
pkill -f python\ exp*
#pkill -f python\ /home/leno/HFO*
killall -9 rcssserver

#Script to run experiments. In its current form, the script is ready to run all the paper experiments.
# It will take very long to complete the experiments, though. If you want the execute some of the experiments, comment out the ones you don't want to execute.
# This script first initiates an instance of the HFO server and then the experiment.py source code is executed with the correct parameters.


#Experiment 1

sh runSimpleExpServer.sh 10000 3 > serverAdHocTD.log &
sleep 5
sh runSimpleExpAgent.sh 10000 AdHocTD 1 50 > logAdHocTD.log 

sh runSimpleExpServer.sh 10100 3 > serverAdHocVisit.log &
sleep 5
sh runSimpleExpAgent.sh 10100 AdHocVisit 1 50 > logAdHocVisit.log 

sh runSimpleExpServer.sh 10200 3 > serverTorrey.log &
sleep 5
sh runSimpleExpAgent.sh 10200 Torrey 1 50 > logTorrey.log 

sh runSimpleExpServer.sh 10300 3 > serverEpisodeSharing.log &
sleep 5
sh runSimpleExpAgent.sh 10300 EpisodeSharing 1 50 > logEpisodeSharing.log 

sh runSimpleExpServer.sh 10400 3 > serverSARSATile.log &
sleep 5
sh runSimpleExpAgent.sh 10400 SARSATile 1 50 > logSARSATile.log 

sh runSimpleExpServer.sh 10000 3 > serverRandom.log &
sleep 5
sh runSimpleExpAgent.sh 10000 Dummy 1 50 > logRandom.log 

#Experiment 2

sh runSimpleExpServer.sh 12000 3 > serverAdHocTDAction.log &
sleep 5
sh runSimpleExpAgent.sh 12000 AdHocTDAction 1 50 > logAdHocTDAction.log 

sh runSimpleExpServer.sh 12100 3 > serverAdHocVisitAction.log &
sleep 5
sh runSimpleExpAgent.sh 12100 AdHocVisitAction 1 50 > logAdHocVisitAction.log 

sh runSimpleExpServer.sh 12200 3 > serverTorreyAction.log &
sleep 5
sh runSimpleExpAgent.sh 12200 TorreyAction 1 50 > logTorreyAction.log 

sh runSimpleExpServer.sh 12300 3 > serverEpisodeSharingAction.log &
sleep 5
sh runSimpleExpAgent.sh 12300 EpisodeSharingAction 1 50 > logEpisodeSharingAction.log 

#Find Experiment 3 in experiment3.sh
