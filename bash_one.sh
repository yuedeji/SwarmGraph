graph_home=/home/yuedeji/small_graph/synthetic_million

make clean
make
for i in 1 10 100 1000 10000 100000 1000000
do
    ./minions $graph_home/graph_group/rmat_graph_1m_random.txt $i > result_$i.csv
done

