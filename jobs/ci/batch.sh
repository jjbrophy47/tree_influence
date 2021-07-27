tt='lgb'

da='adult'
dbm='bank_marketing'
dc='casp'
dco='compas'
dcc='credit_card'
dd='diabetes'
dfd='flight_delays'
dgc='german_credit'
dht='htru2'
dl='life'
dns='no_show'
do='obesity'
dsb='spambase'
ds='surgical'
dtw='twitter'
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

lon='normal'  # local_op
log='sign'  # BoostIn
los='sim'  # BoostIn

gos='self'  # global_op
goe='expected'  # TREX, LOO, and DShap
goa='alpha'  # TREX only

# adult
./jobs/ci/primer.sh     $da $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $da $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $da $tt $ms  $tf $us1 $iol $lon $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $da $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $da $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $da $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iol $lon $gos 6  60    $ps  # trex
# ./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iol $lon $goe 15 60    $ps
# ./jobs/ci/primer.sh     $da $tt $mtx $tf $us1 $iol $lon $goa 6  60    $ps
# ./jobs/ci/primer.sh     $da $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $da $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $da $tt $mlo $tf $us1 $iol $lon $gos 28 600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $da $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $da $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $da $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# bank_marketing
./jobs/ci/primer.sh     $dbm $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dbm $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dbm $tt $ms  $tf $us1 $iol $lon $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dbm $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dbm $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dbm $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iol $lon $goe 6  60    $ps
# ./jobs/ci/primer.sh     $dbm $tt $mtx $tf $us1 $iol $lon $goa 3  60    $ps
# ./jobs/ci/primer.sh     $dbm $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dbm $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dbm $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dbm $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dbm $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dbm $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# casp
./jobs/ci/primer.sh     $dc $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dc $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dc $tt $ms  $tf $us1 $iol $lon $gos 17 60    $ps  # similarity
./jobs/ci/primer.sh     $dc $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dc $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dc $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dc $tt $mtx $tf $us1 $iol $lon $gos 15 60    $ps  # trex
# ./jobs/ci/primer.sh     $dc $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dc $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dc $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dc $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dc $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dc $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dc $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dc $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# compas
./jobs/ci/primer.sh     $dco $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dco $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dco $tt $ms  $tf $us1 $iol $lon $gos 17 60    $ps  # similarity
./jobs/ci/primer.sh     $dco $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dco $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dco $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dco $tt $mtx $tf $us1 $iol $lon $gos 7  60    $ps  # trex
# ./jobs/ci/primer.sh     $dco $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dco $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dco $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dco $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dco $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dco $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dco $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dco $tt $mds $tf $us1 $iol  $lon $goe 28 1440  $ps

# credit_card
./jobs/ci/primer.sh     $dcc $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dcc $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dcc $tt $ms  $tf $us1 $iol $lon $gos 17 60    $ps  # similarity
./jobs/ci/primer.sh     $dcc $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dcc $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dcc $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iol $lon $gos 7  60    $ps  # trex
# ./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dcc $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dcc $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dcc $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dcc $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dcc $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dcc $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dcc $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# diabetes
./jobs/ci/primer.sh     $dd $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dd $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dd $tt $ms  $tf $us1 $iol $lon $gos 15 60    $ps  # similarity
./jobs/ci/primer.sh     $dd $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dd $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dd $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iol $lon $gos 15 60    $ps  # trex
# ./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dd $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dd $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dd $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iol $lon $gos 28 1440  $ps  # loo
# ./jobs/ci/primer_mcu.sh $dd $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dd $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# flight_delays
./jobs/ci/primer.sh     $dfd $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dfd $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dfd $tt $ms  $tf $us1 $iol $lon $gos 15 60    $ps  # similarity
./jobs/ci/primer.sh     $dfd $tt $mbi $tf $us1 $iol $lon $gos 7  60    $ps  # boostin
./jobs/ci/primer.sh     $dfd $tt $mbi $tf $us1 $iol $log $gos 7  60    $ps
./jobs/ci/primer.sh     $dfd $tt $mbi $tf $us1 $iol $los $gos 7  60    $ps
./jobs/ci/primer.sh     $dfd $tt $mtx $tf $us1 $iol $lon $gos 15 60    $ps  # trex
# ./jobs/ci/primer.sh     $dfd $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dfd $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dfd $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dfd $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $dfd $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dfd $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dfd $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dfd $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# german_credit
./jobs/ci/primer.sh     $dgc $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dgc $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dgc $tt $ms  $tf $us1 $iol $lon $gos 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dgc $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dgc $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dgc $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dgc $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $dgc $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dgc $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dgc $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dgc $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dgc $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dgc $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dgc $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dgc $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# htru2
./jobs/ci/primer.sh     $dht $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dht $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dht $tt $ms  $tf $us1 $iol $lon $gos 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dht $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dht $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dht $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dht $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $dht $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dht $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dht $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dht $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dht $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dht $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dht $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dht $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# life
./jobs/ci/primer.sh     $dl $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dl $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dl $tt $ms  $tf $us1 $iol $lon $gos 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dl $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dl $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dl $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dl $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $dl $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dl $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dl $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dl $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dl $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dl $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dl $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dl $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# no_show
./jobs/ci/primer.sh     $dns $tt $mr  $tf $us1 $iol $lon $gos 7  600    $ps  # random
./jobs/ci/primer.sh     $dns $tt $mm  $tf $us1 $iol $lon $gos 7  600    $ps  # minority
./jobs/ci/primer.sh     $dns $tt $ms  $tf $us1 $iol $lon $gos 7  600    $ps  # similarity
./jobs/ci/primer.sh     $dns $tt $mbi $tf $us1 $iol $lon $gos 7  600    $ps  # boostin
./jobs/ci/primer.sh     $dns $tt $mbi $tf $us1 $iol $log $gos 7  600    $ps
./jobs/ci/primer.sh     $dns $tt $mbi $tf $us1 $iol $los $gos 7  600    $ps
./jobs/ci/primer.sh     $dns $tt $mtx $tf $us1 $iol $lon $gos 7  600    $ps  # trex
# ./jobs/ci/primer.sh     $dns $tt $mtx $tf $us1 $iol $lon $goe 70 600    $ps
# ./jobs/ci/primer.sh     $dns $tt $mtx $tf $us1 $iol $lon $goa 15 600    $ps
# ./jobs/ci/primer.sh     $dns $tt $mli $tf $us1 $iol $lon $gos 7  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dns $tt $mli $tf $us0 $iol $lon $gos 7  1440  $ps
./jobs/ci/primer_mcu.sh $dns $tt $mlo $tf $us1 $iol $lon $gos 5  1440  $ps  # loo
# ./jobs/ci/primer_mcu.sh $dns $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dns $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dns $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# obesity
./jobs/ci/primer.sh     $do $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $do $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $do $tt $ms  $tf $us1 $iol $lon $gos 7  60    $ps  # similarity
./jobs/ci/primer.sh     $do $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $do $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $do $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $do $tt $mtx $tf $us1 $iol $lon $gos 15 600   $ps  # trex
# ./jobs/ci/primer.sh     $do $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $do $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $do $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $do $tt $mli $tf $us0 $iol $lon $gos 6  1440  $ps
./jobs/ci/primer_mcu.sh $do $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $do $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $do $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $do $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# spambase
./jobs/ci/primer.sh     $dsb $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dsb $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dsb $tt $ms  $tf $us1 $iol $lon $gos 7  60    $ps  # similarity
./jobs/ci/primer.sh     $dsb $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dsb $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dsb $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dsb $tt $mtx $tf $us1 $iol $lon $gos 7  60    $ps  # trex
# ./jobs/ci/primer.sh     $dsb $tt $mtx $tf $us1 $iol $lon $goe 30 60    $ps
# ./jobs/ci/primer.sh     $dsb $tt $mtx $tf $us1 $iol $lon $goa 15 60    $ps
# ./jobs/ci/primer.sh     $dsb $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dsb $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dsb $tt $mlo $tf $us1 $iol $lon $gos 5  600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dsb $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dsb $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dsb $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# surgical
./jobs/ci/primer.sh     $ds $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $ds $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $ds $tt $ms  $tf $us1 $iol $lon $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $ds $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $ds $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $ds $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iol $lon $goe 6  60    $ps
# ./jobs/ci/primer.sh     $ds $tt $mtx $tf $us1 $iol $lon $goa 3  60    $ps
# ./jobs/ci/primer.sh     $ds $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $ds $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $ds $tt $mlo $tf $us1 $iol $lon $gos 28 600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $ds $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $ds $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $ds $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# twitter
./jobs/ci/primer.sh     $dtw $tt $mr  $tf $us1 $iol $lon $gos 11  600    $ps  # random
./jobs/ci/primer.sh     $dtw $tt $mm  $tf $us1 $iol $lon $gos 11  600    $ps  # minority
./jobs/ci/primer.sh     $dtw $tt $ms  $tf $us1 $iol $lon $goe 11  600    $ps  # similarity
./jobs/ci/primer.sh     $dtw $tt $mbi $tf $us1 $iol $lon $gos 11  600    $ps  # boostin
./jobs/ci/primer.sh     $dtw $tt $mbi $tf $us1 $iol $log $gos 11  600    $ps
./jobs/ci/primer.sh     $dtw $tt $mbi $tf $us1 $iol $los $gos 11  600    $ps
./jobs/ci/primer.sh     $dtw $tt $mtx $tf $us1 $iol $lon $gos 21  600    $ps  # trex
# ./jobs/ci/primer.sh     $dtw $tt $mtx $tf $us1 $iol $lon $goe 6  600    $ps
# ./jobs/ci/primer.sh     $dtw $tt $mtx $tf $us1 $iol $lon $goa 11  600    $ps
# ./jobs/ci/primer.sh     $dtw $tt $mli $tf $us1 $iol $lon $gos 11  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dtw $tt $mli $tf $us0 $iol $lon $gos 11  1440  $ps
./jobs/ci/primer_mcu.sh $dtw $tt $mlo $tf $us1 $iol $lon $gos 28  1440  $ps  # loo
# ./jobs/ci/primer_mcu.sh $dtw $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dtw $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dtw $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps

# vaccine
./jobs/ci/primer.sh     $dv $tt $mr  $tf $us1 $iol $lon $gos 3  60    $ps  # random
./jobs/ci/primer.sh     $dv $tt $mm  $tf $us1 $iol $lon $gos 3  60    $ps  # minority
./jobs/ci/primer.sh     $dv $tt $ms  $tf $us1 $iol $lon $goe 3  60    $ps  # similarity
./jobs/ci/primer.sh     $dv $tt $mbi $tf $us1 $iol $lon $gos 3  60    $ps  # boostin
./jobs/ci/primer.sh     $dv $tt $mbi $tf $us1 $iol $log $gos 3  60    $ps
./jobs/ci/primer.sh     $dv $tt $mbi $tf $us1 $iol $los $gos 3  60    $ps
./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iol $lon $gos 3  60    $ps  # trex
# ./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iol $lon $goe 6  60    $ps
# ./jobs/ci/primer.sh     $dv $tt $mtx $tf $us1 $iol $lon $goa 3  60    $ps
# ./jobs/ci/primer.sh     $dv $tt $mli $tf $us1 $iol $lon $gos 3  10080 $p2  # leaf_influence
./jobs/ci/primer.sh     $dv $tt $mli $tf $us0 $iol $lon $gos 3  1440  $ps
./jobs/ci/primer_mcu.sh $dv $tt $mlo $tf $us1 $iol $lon $gos 28 600   $ps  # loo
# ./jobs/ci/primer_mcu.sh $dv $tt $mlo $tf $us1 $iol $lon $goe 28 600   $ps
# ./jobs/ci/primer_mcu.sh $dv $tt $mds $tf $us1 $iol $lon $gos 28 1440  $ps  # dshap
# ./jobs/ci/primer_mcu.sh $dv $tt $mds $tf $us1 $iol $lon $goe 28 1440  $ps
