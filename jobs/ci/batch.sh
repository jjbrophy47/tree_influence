tt='lgb'

da='adult'
dbm='bank_marketing'
dc='casp'
dcc='credit_card'
dd='diabetes'
dfd='flight_delay'
dgc='german_credit'
dl='life'
do='obesity'
dsb='spambase'
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

# credit_card
./jobs/ci/primer.sh     $dcc $tt $mr  $tf $us1 $iol $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dcc $tt $mm  $tf $us1 $iol $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dcc $tt $ms  $tf $us1 $iol $goe 15 60    $ps  # similarity
./jobs/ci/primer.sh     $dcc $tt $mbi $tf $us1 $iol $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iol $gos 15 60    $ps  # trex
# ./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iog $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iog $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dcc $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dcc $tt $mli $tf $us0 $iol $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dcc $tt $mlo $tf $us1 $iol $gos 28 600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dcc $tt $mlo $tf $us1 $iog $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dcc $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dcc $tt $mds $tf $us1 $iog $goe 28 1440  $ps

# diabetes
./jobs/ci/primer.sh     $dd $tt $mr  $tf $us1 $iol $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dd $tt $mm  $tf $us1 $iol $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dd $tt $ms  $tf $us1 $iol $goe 15 60    $ps  # similarity
./jobs/ci/primer.sh     $dd $tt $mbi $tf $us1 $iol $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iol $gos 15 60    $ps  # trex
# ./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iog $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iog $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dd $tt $mli $tf $us1 $iob $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dd $tt $mli $tf $us0 $iol $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iol $gos 28 600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iog $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iob $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iog $goe 28 1440  $ps

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
