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

./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d1 $tt $nt1 $md1 $m1 $tf $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d2 $tt $nt2 $md2 $m1 $tf $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d3 $tt $nt3 $md3 $m1 $tf $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d4 $tt $nt4 $md4 $m1 $tf $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d5 $tt $nt5 $md5 $m1 $tf $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m1 $tf $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m2 $tf $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m4 $tf $go1 $io2 3 60 $p2  # leaf_influence
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m1 $tf $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $d6 $tt $nt6 $md6 $m1 $tf $go1 $io0 3 60 $p1
