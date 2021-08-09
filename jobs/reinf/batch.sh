tt='lgb'

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

lno='normal'  # local_op, boostin
lsg='sign'  # boostin
lsr='sign_tr'  # boostin
lse='sign_te'  # boostin

sc1=0.0 # boostin leaf_scale
sc1=-1.0
sc1=-2.0
sc1=-3.0

gos='self'  # global_op
goe='expected'  # TREX, LOO, and DShap
goa='alpha'  # TREX only

# adult
./jobs/reinf/primer.sh $da $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/reinf/primer.sh $da $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/reinf/primer.sh $da $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/reinf/primer.sh $da $tt $ms  $tf $us1 $iol $lon $gos 28 1440 $ps  # similarity
./jobs/reinf/primer.sh $da $tt $mbi $tf $us1 $iol $lon $gos 28 1440 $ps  # boostin
./jobs/reinf/primer.sh $da $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/reinf/primer.sh $da $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/reinf/primer.sh $da $tt $mbi $tf $us1 $iol $lot $gos 3 120 $ps
./jobs/reinf/primer.sh $da $tt $mbi $tf $us1 $iol $loh $gos 3 120 $ps
./jobs/reinf/primer.sh $da $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/reinf/primer.sh $da $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $da $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/reinf/primer.sh $da $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/reinf/primer.sh $da $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/reinf/primer.sh $da $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/reinf/primer.sh $da $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $da $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/reinf/primer.sh $da $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# bank_marketing
./jobs/reinf/primer.sh $dbm $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/reinf/primer.sh $dbm $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/reinf/primer.sh $dbm $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/reinf/primer.sh $dbm $tt $ms  $tf $us1 $iol $lon $gos 28 1440 $ps  # similarity
./jobs/reinf/primer.sh $dbm $tt $mbi $tf $us1 $iol $lon $gos 28 1440 $ps  # boostin
./jobs/reinf/primer.sh $dbm $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/reinf/primer.sh $dbm $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/reinf/primer.sh $dbm $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/reinf/primer.sh $dbm $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $dbm $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/reinf/primer.sh $dbm $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/reinf/primer.sh $dbm $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/reinf/primer.sh $dbm $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/reinf/primer.sh $dbm $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $dbm $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/reinf/primer.sh $dbm $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# bean
./jobs/reinf/primer.sh $dbn $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dbn $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dbn $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dbn $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dbn $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $dbn $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dbn $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dbn $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dbn $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dbn $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dbn $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dbn $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dbn $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dbn $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dbn $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dbn $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# casp
./jobs/reinf/primer.sh $dc $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/reinf/primer.sh $dc $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/reinf/primer.sh $dc $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/reinf/primer.sh $dc $tt $ms  $tf $us1 $iol $lon $gos 3 300 $ps  # similarity
./jobs/reinf/primer.sh $dc $tt $mbi $tf $us1 $iol $lon $gos 3 300 $ps  # boostin
./jobs/reinf/primer.sh $dc $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/reinf/primer.sh $dc $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/reinf/primer.sh $dc $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/reinf/primer.sh $dc $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dc $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/reinf/primer.sh $dc $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/reinf/primer.sh $dc $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/reinf/primer.sh $dc $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/reinf/primer.sh $dc $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dc $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/reinf/primer.sh $dc $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# compas
./jobs/reinf/primer.sh $dco $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/reinf/primer.sh $dco $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/reinf/primer.sh $dco $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/reinf/primer.sh $dco $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dco $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $dco $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/reinf/primer.sh $dco $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/reinf/primer.sh $dco $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/reinf/primer.sh $dco $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dco $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/reinf/primer.sh $dco $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/reinf/primer.sh $dco $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/reinf/primer.sh $dco $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/reinf/primer.sh $dco $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dco $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/reinf/primer.sh $dco $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# credit_card
./jobs/reinf/primer.sh $dcc $tt $mr  $tf $us1 $iol $lon $gos 3 300 $ps  # random
# ./jobs/reinf/primer.sh $dcc $tt $mm  $tf $us1 $iol $lon $gos 3 300 $ps  # minority
./jobs/reinf/primer.sh $dcc $tt $mtg $tf $us1 $iol $lon $gos 3 300 $ps  # target
./jobs/reinf/primer.sh $dcc $tt $ms  $tf $us1 $iol $lon $gos 3 300 $ps  # similarity
./jobs/reinf/primer.sh $dcc $tt $mbi $tf $us1 $iol $lon $gos 3 300 $ps  # boostin
./jobs/reinf/primer.sh $dcc $tt $mbi $tf $us1 $iol $log $gos 3 300 $ps
./jobs/reinf/primer.sh $dcc $tt $mbi $tf $us1 $iol $los $gos 3 300 $ps
./jobs/reinf/primer.sh $dcc $tt $mtx $tf $us1 $iol $lon $gos 3 300 $ps  # trex
# ./jobs/reinf/primer.sh $dcc $tt $mtx $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dcc $tt $mtx $tf $us1 $iol $lon $goa 3 300 $ps
# ./jobs/reinf/primer.sh $dcc $tt $mli $tf $us1 $iol $lon $gos 3 300 $p2  # leaf_influence
./jobs/reinf/primer.sh $dcc $tt $mli $tf $us0 $iol $lon $gos 3 300 $ps
./jobs/reinf/primer.sh $dcc $tt $mlo $tf $us1 $iol $lon $gos 3 300 $ps  # loo
# ./jobs/reinf/primer.sh $dcc $tt $mlo $tf $us1 $iol $lon $goe 3 300 $ps
# ./jobs/reinf/primer.sh $dcc $tt $mds $tf $us1 $iol $lon $gos 3 300 $ps  # dshap
# ./jobs/reinf/primer.sh $dcc $tt $mds $tf $us1 $iol $lon $goe 3 300 $ps

# diabetes
./jobs/reinf/primer.sh $dd $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dd $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dd $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dd $tt $ms  $tf $us1 $iol $lon $gos 28 1440 $ps  # similarity
./jobs/reinf/primer.sh $dd $tt $mbi $tf $us1 $iol $lon $gos 28 1440 $ps  # boostin
./jobs/reinf/primer.sh $dd $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dd $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dd $tt $mbi $tf $us1 $iol $lot $gos 3 600 $ps
./jobs/reinf/primer.sh $dd $tt $mbi $tf $us1 $iol $loh $gos 3 600 $ps
./jobs/reinf/primer.sh $dd $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dd $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dd $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dd $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dd $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dd $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dd $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dd $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dd $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# flight_delays
./jobs/reinf/primer.sh $dfd $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dfd $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dfd $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dfd $tt $ms  $tf $us1 $iol $lon $gos 28 1440 $ps  # similarity
./jobs/reinf/primer.sh $dfd $tt $mbi $tf $us1 $iol $lon $gos 28 1440 $ps  # boostin
./jobs/reinf/primer.sh $dfd $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dfd $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dfd $tt $mbi $tf $us1 $iol $lot $gos 3 600 $ps
./jobs/reinf/primer.sh $dfd $tt $mbi $tf $us1 $iol $loh $gos 3 600 $ps
./jobs/reinf/primer.sh $dfd $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dfd $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dfd $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dfd $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dfd $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dfd $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dfd $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dfd $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dfd $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# german_credit
./jobs/reinf/primer.sh $dgc $tt $mr  $tf $us1 $sc1 $iol $lon $gos 3 600  $ps  # random
# ./jobs/reinf/primer.sh $dgc $tt $mm  $tf $us1 $sc1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dgc $tt $mtg $tf $us1 $sc1 $iol $lon $gos 3 600  $ps  # target
./jobs/reinf/primer.sh $dgc $tt $ms  $tf $us1 $sc1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3 600  $ps  # boostin
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mtx $tf $us1 $sc1 $iol $lon $gos 3 600  $ps  # trex
# ./jobs/reinf/primer.sh $dgc $tt $mtx $tf $us1 $sc1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dgc $tt $mtx $tf $us1 $sc1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dgc $tt $mli $tf $us1 $sc1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dgc $tt $mli $tf $us0 $sc1 $iol $lon $gos 3 600  $ps
./jobs/reinf/primer.sh $dgc $tt $mlo $tf $us1 $sc1 $iol $lon $gos 3 1440 $ps  # loo
# ./jobs/reinf/primer.sh $dgc $tt $mlo $tf $us1 $sc1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dgc $tt $mds $tf $us1 $sc1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dgc $tt $mds $tf $us1 $sc1 $iol $lon $goe 3 600 $ps

# htru2
./jobs/reinf/primer.sh $dht $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dht $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dht $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dht $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/reinf/primer.sh $dht $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/reinf/primer.sh $dht $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dht $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dht $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dht $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dht $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dht $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dht $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dht $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dht $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dht $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dht $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# life
./jobs/reinf/primer.sh $dl $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dl $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dl $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dl $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dl $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $dl $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dl $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dl $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dl $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dl $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dl $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dl $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dl $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dl $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dl $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dl $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# no_show
./jobs/reinf/primer.sh $dns $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dns $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dns $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dns $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/reinf/primer.sh $dns $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/reinf/primer.sh $dns $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dns $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dns $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dns $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dns $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dns $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dns $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dns $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dns $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dns $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dns $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# obesity
./jobs/reinf/primer.sh $do $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $do $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $do $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $do $tt $ms  $tf $us1 $iol $lon $gos 3 600 $ps  # similarity
./jobs/reinf/primer.sh $do $tt $mbi $tf $us1 $iol $lon $gos 3 600 $ps  # boostin
./jobs/reinf/primer.sh $do $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $do $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $do $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $do $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $do $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $do $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $do $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $do $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $do $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $do $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $do $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# spambase
./jobs/reinf/primer.sh $dsb $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $dsb $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $dsb $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $dsb $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dsb $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $dsb $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $dsb $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $dsb $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $dsb $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dsb $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $dsb $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $dsb $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $dsb $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $dsb $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $dsb $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $dsb $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# surgical
./jobs/reinf/primer.sh $ds $tt $mr  $tf $us1 $iol $lon $gos 3 600 $ps  # random
# ./jobs/reinf/primer.sh $ds $tt $mm  $tf $us1 $iol $lon $gos 3 600 $ps  # minority
./jobs/reinf/primer.sh $ds $tt $mtg $tf $us1 $iol $lon $gos 3 600 $ps  # target
./jobs/reinf/primer.sh $ds $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $ds $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $ds $tt $mbi $tf $us1 $iol $log $gos 3 600 $ps
./jobs/reinf/primer.sh $ds $tt $mbi $tf $us1 $iol $los $gos 3 600 $ps
./jobs/reinf/primer.sh $ds $tt $mbi $tf $us1 $iol $lot $gos 3 600 $ps
./jobs/reinf/primer.sh $ds $tt $mbi $tf $us1 $iol $loh $gos 3 600 $ps
./jobs/reinf/primer.sh $ds $tt $mtx $tf $us1 $iol $lon $gos 3 600 $ps  # trex
# ./jobs/reinf/primer.sh $ds $tt $mtx $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $ds $tt $mtx $tf $us1 $iol $lon $goa 3 600 $ps
# ./jobs/reinf/primer.sh $ds $tt $mli $tf $us1 $iol $lon $gos 3 600 $p2  # leaf_influence
./jobs/reinf/primer.sh $ds $tt $mli $tf $us0 $iol $lon $gos 3 600 $ps
./jobs/reinf/primer.sh $ds $tt $mlo $tf $us1 $iol $lon $gos 3 600 $ps  # loo
# ./jobs/reinf/primer.sh $ds $tt $mlo $tf $us1 $iol $lon $goe 3 600 $ps
# ./jobs/reinf/primer.sh $ds $tt $mds $tf $us1 $iol $lon $gos 3 600 $ps  # dshap
# ./jobs/reinf/primer.sh $ds $tt $mds $tf $us1 $iol $lon $goe 3 600 $ps

# twitter
./jobs/reinf/primer.sh $tw $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/reinf/primer.sh $tw $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/reinf/primer.sh $tw $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/reinf/primer.sh $tw $tt $ms  $tf $us1 $iol $lon $gos 3 120 $ps  # similarity
./jobs/reinf/primer.sh $tw $tt $mbi $tf $us1 $iol $lon $gos 3 120 $ps  # boostin
./jobs/reinf/primer.sh $tw $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/reinf/primer.sh $tw $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/reinf/primer.sh $tw $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/reinf/primer.sh $tw $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $tw $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/reinf/primer.sh $tw $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/reinf/primer.sh $tw $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/reinf/primer.sh $tw $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/reinf/primer.sh $tw $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $tw $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/reinf/primer.sh $tw $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps

# vaccine
./jobs/reinf/primer.sh $dv $tt $mr  $tf $us1 $iol $lon $gos 3 120 $ps  # random
# ./jobs/reinf/primer.sh $dv $tt $mm  $tf $us1 $iol $lon $gos 3 120 $ps  # minority
./jobs/reinf/primer.sh $dv $tt $mtg $tf $us1 $iol $lon $gos 3 120 $ps  # target
./jobs/reinf/primer.sh $dv $tt $ms  $tf $us1 $iol $lon $gos 3 1440 $ps  # similarity
./jobs/reinf/primer.sh $dv $tt $mbi $tf $us1 $iol $lon $gos 3 1440 $ps  # boostin
./jobs/reinf/primer.sh $dv $tt $mbi $tf $us1 $iol $log $gos 3 120 $ps
./jobs/reinf/primer.sh $dv $tt $mbi $tf $us1 $iol $los $gos 3 120 $ps
./jobs/reinf/primer.sh $dv $tt $mbi $tf $us1 $iol $lot $gos 3 120 $ps
./jobs/reinf/primer.sh $dv $tt $mbi $tf $us1 $iol $loh $gos 3 120 $ps
./jobs/reinf/primer.sh $dv $tt $mtx $tf $us1 $iol $lon $gos 3 120 $ps  # trex
# ./jobs/reinf/primer.sh $dv $tt $mtx $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $dv $tt $mtx $tf $us1 $iol $lon $goa 3 120 $ps
# ./jobs/reinf/primer.sh $dv $tt $mli $tf $us1 $iol $lon $gos 3 120 $p2  # leaf_influence
./jobs/reinf/primer.sh $dv $tt $mli $tf $us0 $iol $lon $gos 3 120 $ps
./jobs/reinf/primer.sh $dv $tt $mlo $tf $us1 $iol $lon $gos 3 120 $ps  # loo
# ./jobs/reinf/primer.sh $dv $tt $mlo $tf $us1 $iol $lon $goe 3 120 $ps
# ./jobs/reinf/primer.sh $dv $tt $mds $tf $us1 $iol $lon $gos 3 120 $ps  # dshap
# ./jobs/reinf/primer.sh $dv $tt $mds $tf $us1 $iol $lon $goe 3 120 $ps
