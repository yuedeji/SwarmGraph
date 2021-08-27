graph_home=/home/yuedeji/small_graph/synthetic_million

make clean
make
#./minions $graph_home/1A56.pdb_BasedOnAllAtoms_4.0.txt
#echo ./minions $graph_home/rmat_graph_list.txt 100
#./minions $graph_home/graph_group/rmat_graph_1024.txt 1000
# ./minions $graph_home/graph_group/rmat_graph_1m_random.txt 100000 > result_apsp_100000.csv

# ./minions $graph_home/graph_group/rmat_graph_1m_random.txt 100 > result_bc_100.csv
for i in 1 10 100 1000 10000 100000 1000000
do
    ./minions $graph_home/graph_group/rmat_graph_1m_random.txt $i > result_apsp_$i.csv
done

