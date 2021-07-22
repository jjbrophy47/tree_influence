tt='lgb'

sk=1  # if 1, skip already present results

da='adult'
dbm='bank_marketing'
dc='casp'
dd='diabetes'
dfd='flight_delays'
dl='life'
do='obesity'
ds='surgical'
dv='vaccine'

mr='random'
mm='minority'
mbi='boostin'
mtx='trex'
mli='leaf_influence'
mlo='loo'
mds='dshap'

ps='short'
pl='long'

tf=0.25  # trunc_frac

us0=0  # update set
us1=-1

iog='global'  # inf_obj
iol='local'
iob='both'

gos='self'  # global_op
goe='expected'  # TREX, LOO, and DShap
goa='alpha'  # TREX only

# surgical
./jobs/roar/primer.sh $sk $ds $tt $mr  $tf $us1 $iob $gos 3 120 $ps  # random
./jobs/roar/primer.sh $sk $ds $tt $mm  $tf $us1 $iob $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $ds $tt $mbi $tf $us1 $iob $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iob $gos 3 120 $ps  # trex
./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iog $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $ds $tt $mli $tf $us1 $iob $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $ds $tt $mli $tf $us0 $iob $gos 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mlo $tf $us1 $iob $gos 3 120 $ps  # loo
./jobs/roar/primer.sh $sk $ds $tt $mlo $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mds $tf $us1 $iob $gos 3 120 $ps  # dshap
./jobs/roar/primer.sh $sk $ds $tt $mds $tf $us1 $iog $goe 3 120 $ps

# bank_marketing
./jobs/roar/primer.sh $sk $dbm $tt $mr  $tf $us1 $iob $gos 3 120 $ps  # random
./jobs/roar/primer.sh $sk $dbm $tt $mm  $tf $us1 $iob $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $dbm $tt $mbi $tf $us1 $iob $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iob $gos 3 120 $ps  # trex
./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iog $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbm $tt $mli $tf $us1 $iob $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dbm $tt $mli $tf $us0 $iob $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mlo $tf $us1 $iob $gos 3 120 $ps  # loo
./jobs/roar/primer.sh $sk $dbm $tt $mlo $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mds $tf $us1 $iob $gos 3 120 $ps  # dshap
./jobs/roar/primer.sh $sk $dbm $tt $mds $tf $us1 $iog $goe 3 120 $ps

# vaccine
./jobs/roar/primer.sh $sk $dv $tt $mr  $tf $us1 $iob $gos 3 120 $ps  # random
./jobs/roar/primer.sh $sk $dv $tt $mm  $tf $us1 $iob $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $dv $tt $mbi $tf $us1 $iob $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iob $gos 3 120 $ps  # trex
./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iog $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $dv $tt $mli $tf $us1 $iob $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dv $tt $mli $tf $us0 $iob $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mlo $tf $us1 $iob $gos 3 120 $ps  # loo
./jobs/roar/primer.sh $sk $dv $tt $mlo $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mds $tf $us1 $iob $gos 3 120 $ps  # dshap
./jobs/roar/primer.sh $sk $dv $tt $mds $tf $us1 $iog $goe 3 120 $ps

# adult
./jobs/roar/primer.sh $sk $da $tt $mr  $tf $us1 $iob $gos 3 120 $ps  # random
./jobs/roar/primer.sh $sk $da $tt $mm  $tf $us1 $iob $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $da $tt $mbi $tf $us1 $iob $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iob $gos 3 120 $ps  # trex
./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iog $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $da $tt $mli $tf $us1 $iob $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $da $tt $mli $tf $us0 $iob $gos 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mlo $tf $us1 $iob $gos 3 120 $ps  # loo
./jobs/roar/primer.sh $sk $da $tt $mlo $tf $us1 $iog $goe 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mds $tf $us1 $iob $gos 3 120 $ps  # dshap
./jobs/roar/primer.sh $sk $da $tt $mds $tf $us1 $iog $goe 3 120 $ps

./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m4 $tf $us0 $go1 $io2 3 60 $p1
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d1 $tt $nt1 $md1 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m4 $tf $us0 $go1 $io2 3 60 $p1
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d2 $tt $nt2 $md2 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m4 $tf $us0 $go1 $io2 3 60 $p1
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d3 $tt $nt3 $md3 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m4 $tf $us0 $go1 $io2 3 60 $p1
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d4 $tt $nt4 $md4 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m4 $tf $us0 $go1 $io2 3 60 $p1
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d5 $tt $nt5 $md5 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m4 $tf $us0 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d6 $tt $nt6 $md6 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m4 $tf $us0 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d7 $tt $nt7 $md7 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m4 $tf $us0 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d8 $tt $nt8 $md8 $m6 $tf $us1 $go1 $io0 3 60 $p1

./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m1 $tf $us1 $go1 $io2 3 60 $p1  # random
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m2 $tf $us1 $go1 $io2 3 60 $p1  # boostin
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m3 $tf $us1 $go1 $io2 3 60 $p1  # trex
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m3 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m3 $tf $us1 $go3 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m4 $tf $us1 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m4 $tf $us0 $go1 $io2 3 60 $p1  # leaf_influence
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m5 $tf $us1 $go1 $io2 3 60 $p1  # loo
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m5 $tf $us1 $go2 $io0 3 60 $p1
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m6 $tf $us1 $go1 $io2 3 60 $p1  # dshap
./jobs/roar/primer.sh $sk $d9 $tt $nt9 $md9 $m6 $tf $us1 $go1 $io0 3 60 $p1
