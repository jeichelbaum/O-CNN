
MAX_ITER=100000
FILE="logs/ae/aocnn6_slim2_070/model/iter_320000.ckpt.index"


i="0"
while [ $i -lt $MAX_ITER ]
do
	# max iter while loop break condition
	i=$[$i+1]
	
	if test -f $FILE; then
		echo "training done"
		break
	else
		python run_ae.py configs/ae_octree.yaml
	fi
done

