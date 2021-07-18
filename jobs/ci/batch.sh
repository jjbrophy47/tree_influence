d1='adult'
d2='bank_marketing'
d3='surgical'
d4='vaccine'
d5='casp'
d6='obesity'

nt1=200
nt2=50
nt3=50
nt4=200
nt5=200
nt6=200

md1=5
md2=5
md3=7
md4=3
md5=7
md6=7

m1='random'
m2='boostin'
m3='trex'
m4='leaf_influence'
m5='loo'
m6='dshap'

dataset_list=('adult' 'bank_marketing' 'surgical' 'vaccine' 'casp' 'obesity')
n_tree_list=(200 50 50 200 200 200)
max_depth_list=(5 5 7 3 7 7)

method_list=('random' 'boostin' 'trex' 'leaf_influence' 'loo')
mem_list=(3 3 3 7 7)
time_list=(60 60 60 10080 600)

./jobs/ci/primer.sh $d1 'lgb' $nt1 $md1 $m1 3 60 'short'
./jobs/ci/primer.sh $d1 'lgb' $nt1 $md1 $m2 3 60 'short'
./jobs/ci/primer.sh $d1 'lgb' $nt1 $md1 $m3 3 60 'short'
./jobs/ci/primer.sh $d1 'lgb' $nt1 $md1 $m4 7 10080 'short'
./jobs/ci/primer.sh $d1 'lgb' $nt1 $md1 $m5 7 600 'short'

./jobs/ci/primer.sh $d2 'lgb' $nt2 $md2 $m1 3 60 'short'
./jobs/ci/primer.sh $d2 'lgb' $nt2 $md2 $m2 3 60 'short'
./jobs/ci/primer.sh $d2 'lgb' $nt2 $md2 $m3 3 60 'short'
./jobs/ci/primer.sh $d2 'lgb' $nt2 $md2 $m4 7 2880 'long'
./jobs/ci/primer.sh $d2 'lgb' $nt2 $md2 $m5 7 600 'short'

./jobs/ci/primer.sh $d3 'lgb' $nt3 $md3 $m1 3 60 'short'
./jobs/ci/primer.sh $d3 'lgb' $nt3 $md3 $m2 3 60 'short'
./jobs/ci/primer.sh $d3 'lgb' $nt3 $md3 $m3 3 60 'short'
./jobs/ci/primer.sh $d3 'lgb' $nt3 $md3 $m4 7 600 'long'
./jobs/ci/primer.sh $d3 'lgb' $nt3 $md3 $m5 7 600 'short'
./jobs/ci/primer_multi_cpu.sh $d3 'lgb' $nt3 $md3 $m6 0.25 28 1440 'short'

./jobs/ci/primer.sh 'vaccine' 'lgb' 200 3 'random'         3  60   'short'
./jobs/ci/primer.sh 'vaccine' 'lgb' 200 3 'boostin'        3  60   'short'
./jobs/ci/primer.sh 'vaccine' 'lgb' 200 3 'trex'           3  60   'short'
./jobs/ci/primer.sh 'vaccine' 'lgb' 200 3 'leaf_influence' 7  4320 'short'
./jobs/ci/primer.sh 'vaccine' 'lgb' 200 3 'loo'            7  600  'short'
./jobs/ci/primer_multi_cpu.sh 'surgical' 'lgb' 200 3 'dshap'         0.25  28  1440 'short'

./jobs/ci/primer.sh 'casp' 'lgb' 200 7 'random'         3  60   'short'
./jobs/ci/primer.sh 'casp' 'lgb' 200 7 'boostin'        3  60   'short'
./jobs/ci/primer.sh 'casp' 'lgb' 200 7 'trex'           3  60   'short'
./jobs/ci/primer.sh 'casp' 'lgb' 200 7 'leaf_influence' 7  4320 'short'
./jobs/ci/primer.sh 'casp' 'lgb' 200 7 'loo'            7  600  'short'



./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'random'         7  60  'short'
./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'boostin'        7  60  'short'
./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'trex'           7  60  'short'
./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'leaf_influence' 7  300 'short'
./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'loo'            7  300 'short'

./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'random'         3  60  'short'
./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'boostin'        3  60  'short'
./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'trex'           3  60  'short'
./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'leaf_influence' 7  300 'short'
./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'loo'            7  300 'short'

./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'random'         7  60   'short'
./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'boostin'        7  60   'short'
./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'trex'           7  60   'short'
./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'leaf_influence' 7  1880 'short'
./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'loo'            7  300  'short'
