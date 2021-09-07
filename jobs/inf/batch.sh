tt='lgb'

dad='adult'
dbm='bank_marketing'
dbn='bean'
dco='compas'
dcn='concrete'
dcc='credit_card'
ddb='diabetes'
den='energy'
dfd='flight_delays'
dgc='german_credit'
dht='htru2'
dlf='life'
dnv='naval'
dns='no_show'
dms='msd'
dob='obesity'
dpw='power'
dpr='protein'
dsb='spambase'
dsg='surgical'
dtw='twitter'
dvc='vaccine'
dwn='wine'

mr='random'
mm='minority'
mbi='boostin'
mtx='trex'
mli='leaf_influence'
mls='leaf_influenceSP'
mlo='loo'
mds='dshap'
ms='leaf_sim'
mis='input_similarity'
ml='loss'
mtg='target'
msb='subsample'

ps='short'
pl='long'

tf=0.25  # trunc_frac

us0=0  # leaf_influence update set
us1=-1

ne1=3000  # trex, n_epoch
ne2=200

./jobs/inf/runner.sh --array=1-22 --cpus-per-task=3 --time=1440 --partition='short' 'random' 'lgb'

# adult
./jobs/inf/primer.sh     $dad $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dad $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dad $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dad $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dad $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dad $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dad $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dad $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dad $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dad $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# bank_marketing
./jobs/inf/primer.sh     $dbm $tt $mr  $tf $us1 $ne1 3  600    $ps  # random
./jobs/inf/primer.sh     $dbm $tt $mtg $tf $us1 $ne1 3  600    $ps  # target
./jobs/inf/primer.sh     $dbm $tt $ms  $tf $us1 $ne1 3  600    $ps  # similarity2
./jobs/inf/primer.sh     $dbm $tt $mis $tf $us1 $ne1 3  600    $ps  # input_similarity
./jobs/inf/primer.sh     $dbm $tt $mb2 $tf $us1 $ne1 3  600    $ps  # boostin2
./jobs/inf/primer.sh     $dbm $tt $mls $tf $us0 $ne1 3  600    $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dbm $tt $mtx $tf $us1 $ne1 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dbm $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dbm $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dbm $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# bean
./jobs/inf/primer.sh     $dbn $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dbn $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dbn $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dbn $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dbn $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dbn $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dbn $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dbn $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dbn $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dbn $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# compas
./jobs/inf/primer.sh     $dco $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dco $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dco $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dco $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dco $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dco $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dco $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dco $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dco $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dco $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# concrete
./jobs/inf/primer.sh     $dcn $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dcn $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dcn $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dcn $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dcn $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dcn $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dcn $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dcn $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dcn $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dcn $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# credit_card
./jobs/inf/primer.sh     $dcc $tt $mr  $tf $us1 $ne1 3  600    $ps  # random
./jobs/inf/primer.sh     $dcc $tt $mtg $tf $us1 $ne1 3  600    $ps  # target
./jobs/inf/primer.sh     $dcc $tt $ms  $tf $us1 $ne1 3  600    $ps  # similarity2
./jobs/inf/primer.sh     $dcc $tt $mis $tf $us1 $ne1 3  600    $ps  # input_similarity
./jobs/inf/primer.sh     $dcc $tt $mb2 $tf $us1 $ne1 3  600    $ps  # boostin2
./jobs/inf/primer.sh     $dcc $tt $mls $tf $us0 $ne1 3  600    $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dcc $tt $mtx $tf $us1 $ne1 6  600    $ps  # trex
# ./jobs/inf/primer.sh     $dcc $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dcc $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dcc $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# diabetes
./jobs/inf/primer.sh     $ddb $tt $mr  $tf $us1 $ne1 3  1440    $ps  # random
./jobs/inf/primer.sh     $ddb $tt $mtg $tf $us1 $ne1 3  1440    $ps  # target
./jobs/inf/primer.sh     $ddb $tt $ms  $tf $us1 $ne1 3  1440    $ps  # similarity2
./jobs/inf/primer.sh     $ddb $tt $mis $tf $us1 $ne1 3  1440    $ps  # input_similarity
./jobs/inf/primer.sh     $ddb $tt $mb2 $tf $us1 $ne1 3  1440    $ps  # boostin2
./jobs/inf/primer.sh     $ddb $tt $mls $tf $us0 $ne1 3  1440    $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $ddb $tt $mtx $tf $us1 $ne1 9  1440    $ps  # trex
# ./jobs/inf/primer.sh     $ddb $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $ddb $tt $mlo $tf $us1 $ne1 7  1440  $ps  # loo
./jobs/inf/primer.sh     $ddb $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# energy
./jobs/inf/primer.sh     $den $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $den $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $den $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $den $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $den $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $den $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $den $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $den $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $den $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $den $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# flight_delays
./jobs/inf/primer.sh     $dfd $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dfd $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dfd $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dfd $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dfd $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dfd $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dfd $tt $mtx $tf $us1 $ne1 5  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dfd $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dfd $tt $mlo $tf $us1 $ne1 7  1440  $ps  # loo
./jobs/inf/primer.sh     $dfd $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# german_credit
./jobs/inf/primer.sh     $dgc $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dgc $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dgc $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dgc $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dgc $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dgc $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dgc $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dgc $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dgc $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dgc $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# htru2
./jobs/inf/primer.sh     $dht $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dht $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dht $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dht $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dht $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dht $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dht $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dht $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dht $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dht $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# life
./jobs/inf/primer.sh     $dlf $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dlf $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dlf $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dlf $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dlf $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dlf $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dlf $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dlf $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dlf $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dlf $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# msd
./jobs/inf/primer.sh     $dms $tt $mr  $tf $us1 $ne1 20  1440  $ps  # random
./jobs/inf/primer.sh     $dms $tt $mtg $tf $us1 $ne1 20  1440  $ps  # target
./jobs/inf/primer.sh     $dms $tt $ms  $tf $us1 $ne1 20  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dms $tt $mis $tf $us1 $ne1 20  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dms $tt $mb2 $tf $us1 $ne1 20  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dms $tt $mls $tf $us0 $ne1 20  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dms $tt $mtx $tf $us1 $ne2 28  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dms $tt $mli $tf $us1 $ne1 20  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dms $tt $mlo $tf $us1 $ne1 28  1440  $ps  # loo
./jobs/inf/primer.sh     $dms $tt $msb $tf $us1 $ne1 20  1440  $ps  # subsample

# naval
./jobs/inf/primer.sh     $dnv $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dnv $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dnv $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dnv $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dnv $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dnv $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dnv $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dnv $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dnv $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dnv $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# no_show
./jobs/inf/primer.sh     $dns $tt $mr  $tf $us1 $ne1 3  600   $ps  # random
./jobs/inf/primer.sh     $dns $tt $mtg $tf $us1 $ne1 3  600   $ps  # target
./jobs/inf/primer.sh     $dns $tt $ms  $tf $us1 $ne1 3  600   $ps  # similarity2
./jobs/inf/primer.sh     $dns $tt $mis $tf $us1 $ne1 3  600   $ps  # input_similarity
./jobs/inf/primer.sh     $dns $tt $mb2 $tf $us1 $ne1 3  600   $ps  # boostin2
./jobs/inf/primer.sh     $dns $tt $mls $tf $us0 $ne1 3  600   $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dns $tt $mtx $tf $us1 $ne1 6  600   $ps  # trex
# ./jobs/inf/primer.sh     $dns $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dns $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dns $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# obesity
./jobs/inf/primer.sh     $dob $tt $mr  $tf $us1 $ne1 3  600   $ps  # random
./jobs/inf/primer.sh     $dob $tt $mtg $tf $us1 $ne1 3  600   $ps  # target
./jobs/inf/primer.sh     $dob $tt $ms  $tf $us1 $ne1 3  600   $ps  # similarity2
./jobs/inf/primer.sh     $dob $tt $mis $tf $us1 $ne1 3  600   $ps  # input_similarity
./jobs/inf/primer.sh     $dob $tt $mb2 $tf $us1 $ne1 3  600   $ps  # boostin2
./jobs/inf/primer.sh     $dob $tt $mls $tf $us0 $ne1 3  600   $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dob $tt $mtx $tf $us1 $ne1 6  600   $ps  # trex
# ./jobs/inf/primer.sh     $dob $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dob $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dob $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# power
./jobs/inf/primer.sh     $dpw $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dpw $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dpw $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dpw $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dpw $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dpw $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dpw $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dpw $tt $mli $tf $us1 $ne1 3  1440   $ps  # leaf_influence
./jobs/inf/primer.sh     $dpw $tt $mlo $tf $us1 $ne1 5  1440   $ps  # loo
./jobs/inf/primer.sh     $dpw $tt $msb $tf $us1 $ne1 5  1440   $ps  # subsample

# protein
./jobs/inf/primer.sh     $dpr $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dpr $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dpr $tt $ms  $tf $us1 $ne1 5  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dpr $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dpr $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dpr $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dpr $tt $mtx $tf $us1 $ne1 5  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dpr $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dpr $tt $mlo $tf $us1 $ne1 7  1440  $ps  # loo
./jobs/inf/primer.sh     $dpr $tt $msb $tf $us1 $ne1 7  1440  $ps  # subsample

# spambase
./jobs/inf/primer.sh     $dsb $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dsb $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dsb $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dsb $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dsb $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dsb $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dsb $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dsb $tt $mli $tf $us1 $ne1 3  1440   $pl  # leaf_influence
./jobs/inf/primer.sh     $dsb $tt $mlo $tf $us1 $ne1 5  1440   $ps  # loo
./jobs/inf/primer.sh     $dsb $tt $msb $tf $us1 $ne1 5  1440   $ps  # subsample

# surgical
./jobs/inf/primer.sh     $dsg $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dsg $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dsg $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dsg $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dsg $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dsg $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dsg $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dsg $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dsg $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dsg $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# twitter
./jobs/inf/primer.sh     $dtw $tt $mr  $tf $us1 $ne1 11  1440  $ps  # random
./jobs/inf/primer.sh     $dtw $tt $mtg $tf $us1 $ne1 11  1440  $ps  # target
./jobs/inf/primer.sh     $dtw $tt $ms  $tf $us1 $ne1 11  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dtw $tt $mis $tf $us1 $ne1 11  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dtw $tt $mb2 $tf $us1 $ne1 11  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dtw $tt $mls $tf $us0 $ne1 11  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dtw $tt $mtx $tf $us1 $ne1 28  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dtw $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dtw $tt $mlo $tf $us1 $ne1 28  1440  $ps  # loo
./jobs/inf/primer.sh     $dtw $tt $msb $tf $us1 $ne1 28  1440  $ps  # subsample

# vaccine
./jobs/inf/primer.sh     $dvc $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dvc $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dvc $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dvc $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dvc $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dvc $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dvc $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dvc $tt $mli $tf $us1 $ne1 3  10080 $p2  # leaf_influence
./jobs/inf/primer.sh     $dvc $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dvc $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample

# wine
./jobs/inf/primer.sh     $dwn $tt $mr  $tf $us1 $ne1 3  1440  $ps  # random
./jobs/inf/primer.sh     $dwn $tt $mtg $tf $us1 $ne1 3  1440  $ps  # target
./jobs/inf/primer.sh     $dwn $tt $ms  $tf $us1 $ne1 3  1440  $ps  # similarity2
./jobs/inf/primer.sh     $dwn $tt $mis $tf $us1 $ne1 3  1440  $ps  # input_similarity
./jobs/inf/primer.sh     $dwn $tt $mb2 $tf $us1 $ne1 3  1440  $ps  # boostin2
./jobs/inf/primer.sh     $dwn $tt $mls $tf $us0 $ne1 3  1440  $ps  # leaf_influenceSP
./jobs/inf/primer.sh     $dwn $tt $mtx $tf $us1 $ne1 6  1440  $ps  # trex
# ./jobs/inf/primer.sh     $dwn $tt $mli $tf $us1 $ne1 3  1440  $ps  # leaf_influence
./jobs/inf/primer.sh     $dwn $tt $mlo $tf $us1 $ne1 5  1440  $ps  # loo
./jobs/inf/primer.sh     $dwn $tt $msb $tf $us1 $ne1 5  1440  $ps  # subsample
