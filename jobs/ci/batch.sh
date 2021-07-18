tt='lgb'

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

p1='short'
p2='long'

tf=0.25  # trunc_frac

go1='self'  # global_op
go2='global'  # global_op: TREX, LOO, and DShap
go3='alpha'  # global_op: TREX only\

io0=0  # 0 - global, 1 - local, 2 - both
io1=1
io2=2

./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m4 $tf $go1 $io2 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d1 $tt $nt1 $md1 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d1 $tt $nt1 $md1 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d1 $tt $nt1 $md1 $m6 $tf $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m4 $tf $go1 $io2 3  2880  $p2  # leaf_influence
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d2 $tt $nt2 $md2 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d2 $tt $nt2 $md2 $m6 $tf $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m4 $tf $go1 $io2 3  300   $p2  # leaf_influence
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d3 $tt $nt3 $md3 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d3 $tt $nt3 $md3 $m6 $tf $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m4 $tf $go1 $io2 3  4320  $p2  # leaf_influence
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d4 $tt $nt4 $md4 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d4 $tt $nt4 $md4 $m6 $tf $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m4 $tf $go1 $io2 3  4320  $p2  # leaf_influence
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d5 $tt $nt5 $md5 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d5 $tt $nt5 $md5 $m6 $tf $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m1 $tf $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m2 $tf $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m4 $tf $go1 $io2 3  4320  $p2  # leaf_influence
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d6 $tt $nt6 $md6 $m6 $tf $go1 $io2 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d6 $tt $nt6 $md6 $m6 $tf $go2 $io0 28 1440 $p1



# ./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'random'         7  60  'short'
# ./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'boostin'        7  60  'short'
# ./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'trex'           7  60  'short'
# ./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'leaf_influence' 7  300 'short'
# ./jobs/ci/primer.sh 'synth_binary' 'lgb' 100 7 'loo'            7  300 'short'

# ./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'random'         3  60  'short'
# ./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'boostin'        3  60  'short'
# ./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'trex'           3  60  'short'
# ./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'leaf_influence' 7  300 'short'
# ./jobs/ci/primer.sh 'synth_regression' 'lgb' 10 2 'loo'            7  300 'short'

# ./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'random'         7  60   'short'
# ./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'boostin'        7  60   'short'
# ./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'trex'           7  60   'short'
# ./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'leaf_influence' 7  1880 'short'
# ./jobs/ci/primer.sh 'synth_multiclass' 'lgb' 200 7 'loo'            7  300  'short'
