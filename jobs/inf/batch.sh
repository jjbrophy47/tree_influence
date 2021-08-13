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

us0=0  # leaf_influence update set
us1=-1

sc0=0.0 # boostin leaf_scale
sc1=-1.0
sc2=-2.0
sc3=-3.0

iog='global'  # inf_obj
iol='local'
iob='both'

lno='normal'  # local_op, boostin
lsg='sign'  # boostin
lsr='sign_tr'  # boostin
lse='sign_te'  # boostin

gos='self'  # global_op
goe='expected'  # TREX, LOO, and DShap
goa='alpha'  # TREX only

# adult
./jobs/inf/primer.sh     $da $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $da $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $da $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $da $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $da $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $da $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $da $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $da $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $da $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $da $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $da $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $da $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $da $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $da $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# bank_marketing
./jobs/inf/primer.sh     $dbm $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dbm $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dbm $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dbm $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbm $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbm $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dbm $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dbm $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dbm $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dbm $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dbm $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dbm $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dbm $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dbm $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# bean
./jobs/inf/primer.sh     $dbn $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dbn $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dbn $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dbn $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbn $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dbn $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dbn $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dbn $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dbn $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dbn $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dbn $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dbn $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dbn $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dbn $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# casp
./jobs/inf/primer.sh     $dc $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # random
# ./jobs/inf/primer.sh     $dc $tt $mm  $tf $us1 $iol $lno $gos 3  1440    $ps  # minority
./jobs/inf/primer.sh     $dc $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # target
./jobs/inf/primer.sh     $dc $tt $ms  $tf $us1 $sc1 $iol $lno $gos 5  1440    $ps  # similarity
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  1440    $ps  # boostin
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dc $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dc $tt $mtx $tf $us1 $sc1 $iol $lno $gos 5  1440    $ps  # trex
# ./jobs/inf/primer.sh     $dc $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  1440    $ps
# ./jobs/inf/primer.sh     $dc $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  1440    $ps
# ./jobs/inf/primer.sh     $dc $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dc $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dc $tt $mlo $tf $us1 $sc1 $iol $lno $gos 7  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dc $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 1440   $ps
# ./jobs/inf/primer_mcu.sh $dc $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dc $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# compas
./jobs/inf/primer.sh     $dco $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dco $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dco $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dco $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dco $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dco $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
./jobs/inf/primer.sh     $dco $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dco $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dco $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dco $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dco $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dco $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# credit_card
./jobs/inf/primer.sh     $dcc $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dcc $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dcc $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dcc $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dcc $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dcc $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dcc $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dcc $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dcc $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dcc $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dcc $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dcc $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dcc $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dcc $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# diabetes
./jobs/inf/primer.sh     $dd $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # random
# ./jobs/inf/primer.sh     $dd $tt $mm  $tf $us1 $iol $lno $gos 3  1440    $ps  # minority
./jobs/inf/primer.sh     $dd $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # target
./jobs/inf/primer.sh     $dd $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # similarity
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  1440    $ps  # boostin
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dd $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dd $tt $mtx $tf $us1 $sc1 $iol $lno $gos 9  1440    $ps  # trex
# ./jobs/inf/primer.sh     $dd $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  1440    $ps
# ./jobs/inf/primer.sh     $dd $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  1440    $ps
# ./jobs/inf/primer.sh     $dd $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dd $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dd $tt $mlo $tf $us1 $sc1 $iol $lno $gos 7  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dd $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 1440   $ps
# ./jobs/inf/primer_mcu.sh $dd $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dd $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# flight_delays
./jobs/inf/primer.sh     $dfd $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # random
# ./jobs/inf/primer.sh     $dfd $tt $mm  $tf $us1 $iol $lno $gos 3  1440    $ps  # minority
./jobs/inf/primer.sh     $dfd $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # target
./jobs/inf/primer.sh     $dfd $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps  # similarity
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  1440    $ps  # boostin
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  1440    $ps
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dfd $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  1440    $ps
./jobs/inf/primer.sh     $dfd $tt $mtx $tf $us1 $sc1 $iol $lno $gos 5  1440    $ps  # trex
# ./jobs/inf/primer.sh     $dfd $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  1440    $ps
# ./jobs/inf/primer.sh     $dfd $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  1440    $ps
# ./jobs/inf/primer.sh     $dfd $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dfd $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dfd $tt $mlo $tf $us1 $sc1 $iol $lno $gos 7  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dfd $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 1440   $ps
# ./jobs/inf/primer_mcu.sh $dfd $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dfd $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# german_credit
./jobs/inf/primer.sh     $dgc $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dgc $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dgc $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dgc $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dgc $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dgc $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
./jobs/inf/primer.sh     $dgc $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dgc $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dgc $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dgc $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dgc $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dgc $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# htru2
./jobs/inf/primer.sh     $dht $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dht $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dht $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dht $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dht $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dht $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dht $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dht $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dht $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dht $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dht $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dht $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dht $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dht $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# life
./jobs/inf/primer.sh     $dl $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dl $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dl $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dl $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dl $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dl $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
./jobs/inf/primer.sh     $dl $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dl $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dl $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dl $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dl $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dl $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# no_show
./jobs/inf/primer.sh     $dns $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dns $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dns $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dns $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dns $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dns $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dns $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dns $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dns $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dns $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dns $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dns $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dns $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dns $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# obesity
./jobs/inf/primer.sh     $do $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $do $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $do $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $do $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $do $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $do $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $do $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $do $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $do $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $do $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $do $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $do $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $do $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $do $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# spambase
./jobs/inf/primer.sh     $dsb $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dsb $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dsb $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dsb $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dsb $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dsb $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
./jobs/inf/primer.sh     $dsb $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  1440  $pl  # leaf_influence
./jobs/inf/primer.sh     $dsb $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dsb $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dsb $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dsb $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dsb $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# surgical
./jobs/inf/primer.sh     $ds $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $ds $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $ds $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $ds $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $ds $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $ds $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $ds $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $ds $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $ds $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $ds $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $ds $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $ds $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $ds $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $ds $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# twitter
./jobs/inf/primer.sh     $dtw $tt $mr  $tf $us1 $sc1 $iol $lno $gos 11  1440    $ps  # random
# ./jobs/inf/primer.sh     $dtw $tt $mm  $tf $us1 $iol $lno $gos 3  1440    $ps  # minority
./jobs/inf/primer.sh     $dtw $tt $mtg $tf $us1 $sc1 $iol $lno $gos 11  1440    $ps  # target
./jobs/inf/primer.sh     $dtw $tt $ms  $tf $us1 $sc1 $iol $lno $gos 11  1440    $ps  # similarity
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc0 $iol $lno $gos 11  1440    $ps  # boostin
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc1 $iol $lno $gos 11  1440    $ps
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc2 $iol $lno $gos 11  1440    $ps
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 11  1440    $ps
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 11  1440    $ps
./jobs/inf/primer.sh     $dtw $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 11  1440    $ps
./jobs/inf/primer.sh     $dtw $tt $mtx $tf $us1 $sc1 $iol $lno $gos 28  1440   $ps  # trex
# ./jobs/inf/primer.sh     $dtw $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dtw $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dtw $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dtw $tt $mli $tf $us0 $sc1 $iol $lno $gos 28 1440  $ps
./jobs/inf/primer.sh     $dtw $tt $mlo $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dtw $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dtw $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dtw $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps

# vaccine
./jobs/inf/primer.sh     $dv $tt $mr  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # random
# ./jobs/inf/primer.sh     $dv $tt $mm  $tf $us1 $iol $lno $gos 3  600    $ps  # minority
./jobs/inf/primer.sh     $dv $tt $mtg $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # target
./jobs/inf/primer.sh     $dv $tt $ms  $tf $us1 $sc1 $iol $lno $gos 3  600    $ps  # similarity
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc0 $iol $lno $gos 3  600    $ps  # boostin
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc1 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc2 $iol $lno $gos 3  600    $ps
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc0 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc1 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dv $tt $mbi $tf $us1 $sc2 $iol $lsg $gos 3  600    $ps
./jobs/inf/primer.sh     $dv $tt $mtx $tf $us1 $sc1 $iol $lno $gos 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dv $tt $mtx $tf $us1 $sc1 $iol $lno $goe 6  600    $ps
# ./jobs/inf/primer.sh     $dv $tt $mtx $tf $us1 $sc1 $iol $lno $goa 3  600    $ps
# ./jobs/inf/primer.sh     $dv $tt $mli $tf $us1 $sc1 $iol $lno $gos 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dv $tt $mli $tf $us0 $sc1 $iol $lno $gos 3  1440  $ps
./jobs/inf/primer.sh     $dv $tt $mlo $tf $us1 $sc1 $iol $lno $gos 5  1440  $ps  # loo
# ./jobs/inf/primer_mcu.sh $dv $tt $mlo $tf $us1 $sc1 $iol $lno $goe 28 600   $ps
# ./jobs/inf/primer_mcu.sh $dv $tt $mds $tf $us1 $sc1 $iol $lno $gos 28 1440  $ps  # dshap
# ./jobs/inf/primer_mcu.sh $dv $tt $mds $tf $us1 $sc1 $iol $lno $goe 28 1440  $ps
