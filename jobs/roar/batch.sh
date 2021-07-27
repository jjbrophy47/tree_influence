tt='lgb'

sk=1  # if 1, skip already present results

da='adult'
dbm='bank_marketing'
dbn='bean'
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
mtg='target'

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
./jobs/roar/primer.sh $sk $da $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $da $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $da $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $da $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $da $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $da $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $da $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $da $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $da $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $da $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $da $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $da $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $da $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# bank_marketing
./jobs/roar/primer.sh $sk $dbm $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $dbm $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $dbm $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $dbm $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $dbm $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $dbm $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbm $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbm $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dbm $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbm $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $dbm $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbm $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dbm $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# bean
./jobs/roar/primer.sh $sk $dbn $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $dbn $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $dbn $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $dbn $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $dbn $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $dbn $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbn $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbn $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $dbn $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbn $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbn $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dbn $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dbn $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $dbn $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dbn $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dbn $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# casp
./jobs/roar/primer.sh $sk $dc $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/roar/primer.sh $sk $dc $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/roar/primer.sh $sk $dc $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/roar/primer.sh $sk $dc $tt $ms  $tf $us1 $iol $lon $gos 3 300 $ps  # similarity
./jobs/roar/primer.sh $sk $dc $tt $mbi $tf $us1 $iol $lon $gos 3 300 $ps  # boostin
./jobs/roar/primer.sh $sk $dc $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dc $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dc $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/roar/primer.sh $sk $dc $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dc $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/roar/primer.sh $sk $dc $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dc $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dc $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/roar/primer.sh $sk $dc $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dc $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dc $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# compas
./jobs/roar/primer.sh $sk $dco $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/roar/primer.sh $sk $dco $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/roar/primer.sh $sk $dco $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/roar/primer.sh $sk $dco $tt $ms  $tf $us1 $iol $lon $gos 3 300 $ps  # similarity
./jobs/roar/primer.sh $sk $dco $tt $mbi $tf $us1 $iol $lon $gos 3 300 $ps  # boostin
./jobs/roar/primer.sh $sk $dco $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dco $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dco $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/roar/primer.sh $sk $dco $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dco $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/roar/primer.sh $sk $dco $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dco $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dco $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/roar/primer.sh $sk $dco $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dco $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dco $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# credit_card
./jobs/roar/primer.sh $sk $dcc $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/roar/primer.sh $sk $dcc $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/roar/primer.sh $sk $dcc $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/roar/primer.sh $sk $dcc $tt $ms  $tf $us1 $iol $lon $gos 3 300 $ps  # similarity
./jobs/roar/primer.sh $sk $dcc $tt $mbi $tf $us1 $iol $lon $gos 3 300 $ps  # boostin
./jobs/roar/primer.sh $sk $dcc $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dcc $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dcc $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/roar/primer.sh $sk $dcc $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dcc $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/roar/primer.sh $sk $dcc $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dcc $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/roar/primer.sh $sk $dcc $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/roar/primer.sh $sk $dcc $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/roar/primer.sh $sk $dcc $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dcc $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# diabetes
./jobs/roar/primer.sh $sk $dd $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dd $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dd $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dd $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dd $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dd $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dd $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dd $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dd $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dd $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dd $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dd $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dd $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dd $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dd $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dd $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# flight_delays
./jobs/roar/primer.sh $sk $dfd $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dfd $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dfd $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dfd $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dfd $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dfd $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dfd $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dfd $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dfd $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dfd $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dfd $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dfd $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dfd $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dfd $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dfd $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dfd $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# german_credit
./jobs/roar/primer.sh $sk $dgc $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dgc $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dgc $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dgc $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dgc $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dgc $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dgc $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dgc $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dgc $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dgc $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dgc $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dgc $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dgc $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dgc $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dgc $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dgc $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# htru2
./jobs/roar/primer.sh $sk $dht $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dht $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dht $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dht $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dht $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dht $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dht $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dht $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dht $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dht $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dht $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dht $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dht $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dht $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dht $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dht $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# life
./jobs/roar/primer.sh $sk $dl $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dl $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dl $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dl $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dl $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dl $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dl $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dl $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dl $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dl $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dl $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dl $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dl $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dl $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dl $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dl $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# no_show
./jobs/roar/primer.sh $sk $dns $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dns $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dns $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dns $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dns $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dns $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dns $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dns $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dns $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dns $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dns $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dns $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dns $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dns $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dns $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dns $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# obesity
./jobs/roar/primer.sh $sk $do $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $do $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $do $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $do $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $do $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $do $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $do $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $do $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $do $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $do $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $do $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $do $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $do $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $do $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $do $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $do $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# spambase
./jobs/roar/primer.sh $sk $dsb $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/roar/primer.sh $sk $dsb $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/roar/primer.sh $sk $dsb $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/roar/primer.sh $sk $dsb $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/roar/primer.sh $sk $dsb $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/roar/primer.sh $sk $dsb $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dsb $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dsb $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/roar/primer.sh $sk $dsb $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dsb $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/roar/primer.sh $sk $dsb $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dsb $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/roar/primer.sh $sk $dsb $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/roar/primer.sh $sk $dsb $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/roar/primer.sh $sk $dsb $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dsb $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# surgical
./jobs/roar/primer.sh $sk $ds $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $ds $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $ds $tt $mtg  $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $ds $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $ds $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $ds $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $ds $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $ds $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $ds $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $ds $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $ds $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $ds $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $ds $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# twitter
./jobs/roar/primer.sh $sk $tw $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $tw $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $tw $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $tw $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $tw $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $tw $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $tw $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $tw $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $tw $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $tw $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $tw $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $tw $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $tw $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $tw $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $tw $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $tw $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# vaccine
./jobs/roar/primer.sh $sk $dv $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/roar/primer.sh $sk $dv $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/roar/primer.sh $sk $dv $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/roar/primer.sh $sk $dv $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/roar/primer.sh $sk $dv $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/roar/primer.sh $sk $dv $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dv $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/roar/primer.sh $sk $dv $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/roar/primer.sh $sk $dv $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/roar/primer.sh $sk $dv $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/roar/primer.sh $sk $dv $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/roar/primer.sh $sk $dv $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/roar/primer.sh $sk $dv $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps
