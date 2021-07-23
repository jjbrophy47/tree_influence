tt='lgb'

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
ms='similarity'
ml='loss'

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
./jobs/ci/primer.sh     $ds $tt $mr  $tf $us1 $iob $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $ds $tt $mm  $tf $us1 $iob $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $ds $tt $ms  $tf $us1 $iol $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $ds $tt $mbi $tf $us1 $iob $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iob $gos 3  60    $ps  # trex
./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iog $goe 6  60    $ps
./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iog $goa 3  60    $ps
# ./jobs/ci/primer.sh     $ds $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $ds $tt $mli $tf $us0 $iob $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $ds $tt $mlo $tf $us1 $iob $gos 28 600   $ps  # loo
./jobs/ci/primer_mcu.sh $ds $tt $mlo $tf $us1 $iog $goe 28 600   $ps
./jobs/ci/primer_mcu.sh $ds $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
./jobs/ci/primer_mcu.sh $ds $tt $mds $tf $us1 $iog $goe 28 1440  $ps

# bank_marketing
./jobs/ci/primer.sh     $dbm $tt $mr  $tf $us1 $iob $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dbm $tt $mm  $tf $us1 $iob $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dbm $tt $ms  $tf $us1 $iol $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dbm $tt $mbi $tf $us1 $iob $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iob $gos 3  60    $ps  # trex
./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iog $goe 6  60    $ps
./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iog $goa 3  60    $ps
# ./jobs/ci/primer.sh     $dbm $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dbm $tt $mli $tf $us0 $iob $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dbm $tt $mlo $tf $us1 $iob $gos 28 600   $ps  # loo
./jobs/ci/primer_mcu.sh $dbm $tt $mlo $tf $us1 $iog $goe 28 600   $ps
./jobs/ci/primer_mcu.sh $dbm $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
./jobs/ci/primer_mcu.sh $dbm $tt $mds $tf $us1 $iog $goe 28 1440  $ps

# vaccine
./jobs/ci/primer.sh     $dv $tt $mr  $tf $us1 $iob $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dv $tt $mm  $tf $us1 $iob $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dv $tt $ms  $tf $us1 $iol $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dv $tt $mbi $tf $us1 $iob $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iob $gos 3  60    $ps  # trex
./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iog $goe 6  60    $ps
./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iog $goa 3  60    $ps
# ./jobs/ci/primer.sh     $dv $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dv $tt $mli $tf $us0 $iob $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dv $tt $mlo $tf $us1 $iob $gos 28 600   $ps  # loo
./jobs/ci/primer_mcu.sh $dv $tt $mlo $tf $us1 $iog $goe 28 600   $ps
./jobs/ci/primer_mcu.sh $dv $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
./jobs/ci/primer_mcu.sh $dv $tt $mds $tf $us1 $iog $goe 28 1440  $ps

# adult
./jobs/ci/primer.sh     $da $tt $mr  $tf $us1 $iob $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $da $tt $mm  $tf $us1 $iob $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $da $tt $ms  $tf $us1 $iol $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $da $tt $mbi $tf $us1 $iob $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iob $gos 6  60    $ps  # trex
./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iog $goe 15 60    $ps
./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iog $goa 6  60    $ps
# ./jobs/ci/primer.sh     $da $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $da $tt $mli $tf $us0 $iob $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $da $tt $mlo $tf $us1 $iob $gos 28 600   $ps  # loo
./jobs/ci/primer_mcu.sh $da $tt $mlo $tf $us1 $iog $goe 28 600   $ps
./jobs/ci/primer_mcu.sh $da $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
./jobs/ci/primer_mcu.sh $da $tt $mds $tf $us1 $iog $goe 28 1440  $ps

# diabetes
./jobs/ci/primer.sh     $dd $tt $mr  $tf $us1 $iob $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dd $tt $mm  $tf $us1 $iob $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dd $tt $ms  $tf $us1 $iol $goe 15 60    $ps  # similarity
./jobs/ci/primer.sh     $dd $tt $mbi $tf $us1 $iob $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iob $gos 15 60    $ps  # trex
./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iog $goe 30 60    $ps
./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iog $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dd $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dd $tt $mli $tf $us0 $iob $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iob $gos 28 600   $ps  # loo
./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iog $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iog $goe 28 1440  $ps

./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $us1 $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $us1 $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m3 $tf $us1 $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m4 $tf $us1 $go1 $io2 3  2880  $p2  # leaf_influence
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $us1 $go1 $io2 6  600   $p1  # loo
./jobs/ci/primer.sh $d2 $tt $nt2 $md2 $m5 $tf $us1 $go2 $io0 6  600   $p1
./jobs/ci/primer_multi_cpu.sh $d2 $tt $nt2 $md2 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d2 $tt $nt2 $md2 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $us1 $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $us1 $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m3 $tf $us1 $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m4 $tf $us1 $go1 $io2 3  300   $p2  # leaf_influence
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $us1 $go1 $io2 3  600   $p1  # loo
./jobs/ci/primer.sh $d3 $tt $nt3 $md3 $m5 $tf $us1 $go2 $io0 3  600   $p1
./jobs/ci/primer_multi_cpu.sh $d3 $tt $nt3 $md3 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d3 $tt $nt3 $md3 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $us1 $go1 $io2 3  60    $p1  # trex
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $us1 $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m3 $tf $us1 $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m4 $tf $us1 $go1 $io2 3  4320  $p2  # leaf_influence
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $us1 $go1 $io2 6  600   $p1  # loo
./jobs/ci/primer.sh $d4 $tt $nt4 $md4 $m5 $tf $us1 $go2 $io0 6  600   $p1
./jobs/ci/primer_multi_cpu.sh $d4 $tt $nt4 $md4 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d4 $tt $nt4 $md4 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $us1 $go1 $io2 6  60    $p1  # trex
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $us1 $go2 $io0 15 60    $p1
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m3 $tf $us1 $go3 $io0 3  60    $p1
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m4 $tf $us1 $go1 $io2 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $us1 $go1 $io2 6  600   $p1  # loo
./jobs/ci/primer.sh $d5 $tt $nt5 $md5 $m5 $tf $us1 $go2 $io0 6  600   $p1
./jobs/ci/primer_multi_cpu.sh $d5 $tt $nt5 $md5 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d5 $tt $nt5 $md5 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $us1 $go1 $io2 6  600   $p1  # trex
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $us1 $go2 $io0 25 600   $p1
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m3 $tf $us1 $go3 $io0 6  600   $p1
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m4 $tf $us1 $go1 $io2 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $us1 $go1 $io2 6  900   $p1  # loo
./jobs/ci/primer.sh $d6 $tt $nt6 $md6 $m5 $tf $us1 $go2 $io0 6  900   $p1
./jobs/ci/primer_multi_cpu.sh $d6 $tt $nt6 $md6 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d6 $tt $nt6 $md6 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m3 $tf $us1 $go1 $io2 6  600   $p1  # trex
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m3 $tf $us1 $go2 $io0 25 600   $p1
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m3 $tf $us1 $go3 $io0 6  600   $p1
# ./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m4 $tf $us1 $go1 $io2 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m5 $tf $us1 $go1 $io2 6  900   $p1  # loo
./jobs/ci/primer.sh $d7 $tt $nt7 $md7 $m5 $tf $us1 $go2 $io0 6  900   $p1
./jobs/ci/primer_multi_cpu.sh $d7 $tt $nt7 $md7 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d7 $tt $nt7 $md7 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m3 $tf $us1 $go1 $io2 6  600   $p1  # trex
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m3 $tf $us1 $go2 $io0 25 600   $p1
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m3 $tf $us1 $go3 $io0 6  600   $p1
# ./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m4 $tf $us1 $go1 $io2 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m5 $tf $us1 $go1 $io2 6  900   $p1  # loo
./jobs/ci/primer.sh $d8 $tt $nt8 $md8 $m5 $tf $us1 $go2 $io0 6  900   $p1
./jobs/ci/primer_multi_cpu.sh $d8 $tt $nt8 $md8 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d8 $tt $nt8 $md8 $m6 $tf $us1 $go2 $io0 28 1440 $p1

./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m1 $tf $us1 $go1 $io2 3  60    $p1  # random
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m2 $tf $us1 $go1 $io2 3  60    $p1  # boostin
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m3 $tf $us1 $go1 $io2 6  600   $p1  # trex
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m3 $tf $us1 $go2 $io0 7  600   $p1
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m3 $tf $us1 $go3 $io0 6  600   $p1
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m4 $tf $us1 $go1 $io2 3  2880  $p2  # leaf_influence
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m4 $tf $us0 $go1 $io2 3  1440  $p1
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m5 $tf $us1 $go1 $io2 6  900   $p1  # loo
./jobs/ci/primer.sh $d9 $tt $nt9 $md9 $m5 $tf $us1 $go2 $io0 6  900   $p1
./jobs/ci/primer_multi_cpu.sh $d9 $tt $nt9 $md9 $m6 $tf $us1 $go1 $io1 28 1440 $p1  # dshap
./jobs/ci/primer_multi_cpu.sh $d9 $tt $nt9 $md9 $m6 $tf $us1 $go2 $io0 28 1440 $p1
