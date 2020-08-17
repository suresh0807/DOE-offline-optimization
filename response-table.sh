echo "X L1 L2 L3"
for i in $(seq 1 9)
do
        awk -v ii=$i 'NR>2{print $ii" "$10}' set1 | sort -n -k 1 | split --lines=9
        avg1=`average.sh 2 xaa| awk 'NR==1{print $2}'`
        avg2=`average.sh 2 xab| awk 'NR==1{print $2}'`
        avg3=`average.sh 2 xac| awk 'NR==1{print $2}'`
        echo "X$i $avg1 $avg2 $avg3"
done
