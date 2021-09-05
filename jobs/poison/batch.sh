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
mb2='boostin2'
mb3='boostin3'
mb4='boostin4'
mtx='trex'
mli='leaf_influence'
mls='leaf_influenceSP'
mlo='loo'
mds='dshap'
ms='similarity2'
mis='input_similarity'
ml='loss'
mtg='target'
msb='subsample'

ps='short'
pl='long'

tf=0.25  # trunc_frac

us0=0  # leaf_influence update set
us1=-1

# adult
./jobs/poison/primer.sh     $dad $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dad $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dad $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dad $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dad $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dad $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dad $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dad $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dad $tt $msb $tf $us1 5  1440  $ps  # subsample

# bank_marketing
./jobs/poison/primer.sh     $dbm $tt $mr  $tf $us1 3  600   $ps  # random
./jobs/poison/primer.sh     $dbm $tt $ms  $tf $us1 3  600   $ps  # similarity2
./jobs/poison/primer.sh     $dbm $tt $mis $tf $us1 3  600   $ps  # input_similarity
./jobs/poison/primer.sh     $dbm $tt $mb2 $tf $us1 3  600   $ps  # boostin2
./jobs/poison/primer.sh     $dbm $tt $mls $tf $us0 3  600   $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dbm $tt $mtx $tf $us1 6  600   $ps  # trex
# ./jobs/poison/primer.sh     $dbm $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dbm $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dbm $tt $msb $tf $us1 5  1440  $ps  # subsample

# bean
./jobs/poison/primer.sh     $dbn $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dbn $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dbn $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dbn $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dbn $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dbn $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dbn $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dbn $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dbn $tt $msb $tf $us1 5  1440  $ps  # subsample

# compas
./jobs/poison/primer.sh     $dco $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dco $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dco $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dco $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dco $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dco $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dco $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dco $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dco $tt $msb $tf $us1 5  1440  $ps  # subsample

# concrete
./jobs/poison/primer.sh     $dcn $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dcn $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dcn $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dcn $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dcn $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dcn $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dcn $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dcn $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dcn $tt $msb $tf $us1 5  1440  $ps  # subsample

# credit_card
./jobs/poison/primer.sh     $dcc $tt $mr  $tf $us1 3  600   $ps  # random
./jobs/poison/primer.sh     $dcc $tt $ms  $tf $us1 3  600   $ps  # similarity2
./jobs/poison/primer.sh     $dcc $tt $mis $tf $us1 3  600   $ps  # input_similarity
./jobs/poison/primer.sh     $dcc $tt $mb2 $tf $us1 3  600   $ps  # boostin2
./jobs/poison/primer.sh     $dcc $tt $mls $tf $us0 3  600   $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dcc $tt $mtx $tf $us1 6  600   $ps  # trex
# ./jobs/poison/primer.sh     $dcc $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dcc $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dcc $tt $msb $tf $us1 5  1440  $ps  # subsample

# diabetes
./jobs/poison/primer.sh     $ddb $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $ddb $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $ddb $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $ddb $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $ddb $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $ddb $tt $mtx $tf $us1 9  1440  $ps  # trex
# ./jobs/poison/primer.sh     $ddb $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $ddb $tt $mlo $tf $us1 7  1440  $ps  # loo
./jobs/poison/primer.sh     $ddb $tt $msb $tf $us1 5  1440  $ps  # subsample

# energy
./jobs/poison/primer.sh     $den $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $den $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $den $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $den $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $den $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $den $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $den $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $den $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $den $tt $msb $tf $us1 5  1440  $ps  # subsample

# flight_delays
./jobs/poison/primer.sh     $dfd $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dfd $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dfd $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dfd $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dfd $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dfd $tt $mtx $tf $us1 5  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dfd $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dfd $tt $mlo $tf $us1 7  1440  $ps  # loo
./jobs/poison/primer.sh     $dfd $tt $msb $tf $us1 5  1440  $ps  # subsample

# german_credit
./jobs/poison/primer.sh     $dgc $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dgc $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dgc $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dgc $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dgc $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dgc $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dgc $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dgc $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dgc $tt $msb $tf $us1 5  1440  $ps  # subsample

# htru2
./jobs/poison/primer.sh     $dht $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dht $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dht $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dht $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dht $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dht $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dht $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dht $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dht $tt $msb $tf $us1 5  1440  $ps  # subsample

# life
./jobs/poison/primer.sh     $dlf $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dlf $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dlf $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dlf $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dlf $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dlf $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dlf $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dlf $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dlf $tt $msb $tf $us1 5  1440  $ps  # subsample

# msd
# ./jobs/poison/primer.sh     $dms $tt $mr  $tf $us1 20  1440  $ps  # random
# ./jobs/poison/primer.sh     $dms $tt $ms  $tf $us1 20  1440  $ps  # similarity2
# ./jobs/poison/primer.sh     $dms $tt $mis $tf $us1 20  1440  $ps  # input_similarity
# ./jobs/poison/primer.sh     $dms $tt $mb2 $tf $us1 20  1440  $ps  # boostin2
# ./jobs/poison/primer.sh     $dms $tt $mls $tf $us0 20  1440  $ps  # leaf_influenceSP
# ./jobs/poison/primer.sh     $dms $tt $mtx $tf $us1 28  1440  $ps  # trex
# # ./jobs/poison/primer.sh     $dms $tt $mli $tf $us1 20  1440  $ps  # leaf_influence
# ./jobs/poison/primer.sh     $dms $tt $mlo $tf $us1 28  1440  $ps  # loo
# ./jobs/poison/primer.sh     $dms $tt $msb $tf $us1 20  1440  $ps  # subsample

# naval
./jobs/poison/primer.sh     $dnv $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dnv $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dnv $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dnv $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dnv $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dnv $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dnv $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dnv $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dnv $tt $msb $tf $us1 5  1440  $ps  # subsample

# no_show
./jobs/poison/primer.sh     $dns $tt $mr  $tf $us1 3  600   $ps  # random
./jobs/poison/primer.sh     $dns $tt $ms  $tf $us1 3  600   $ps  # similarity2
./jobs/poison/primer.sh     $dns $tt $mis $tf $us1 3  600   $ps  # input_similarity
./jobs/poison/primer.sh     $dns $tt $mb2 $tf $us1 3  600   $ps  # boostin2
./jobs/poison/primer.sh     $dns $tt $mls $tf $us0 3  600   $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dns $tt $mtx $tf $us1 6  600   $ps  # trex
# ./jobs/poison/primer.sh     $dns $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dns $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dns $tt $msb $tf $us1 5  1440  $ps  # subsample

# obesity
./jobs/poison/primer.sh     $dob $tt $mr  $tf $us1 3  600   $ps  # random
./jobs/poison/primer.sh     $dob $tt $ms  $tf $us1 3  600   $ps  # similarity2
./jobs/poison/primer.sh     $dob $tt $mis $tf $us1 3  600   $ps  # input_similarity
./jobs/poison/primer.sh     $dob $tt $mb2 $tf $us1 3  600   $ps  # boostin2
./jobs/poison/primer.sh     $dob $tt $mls $tf $us0 3  600   $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dob $tt $mtx $tf $us1 6  600   $ps  # trex
# ./jobs/poison/primer.sh     $dob $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dob $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dob $tt $msb $tf $us1 5  1440  $ps  # subsample

# power
./jobs/poison/primer.sh     $dpw $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dpw $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dpw $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dpw $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dpw $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dpw $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dpw $tt $mli $tf $us1 3  1440   $ps  # leaf_influence
./jobs/poison/primer.sh     $dpw $tt $mlo $tf $us1 5  1440   $ps  # loo
./jobs/poison/primer.sh     $dpw $tt $msb $tf $us1 5  1440   $ps  # subsample

# protein
./jobs/poison/primer.sh     $dpr $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dpr $tt $ms  $tf $us1 5  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dpr $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dpr $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dpr $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dpr $tt $mtx $tf $us1 5  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dpr $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dpr $tt $mlo $tf $us1 7  1440  $ps  # loo
./jobs/poison/primer.sh     $dpr $tt $msb $tf $us1 7  1440  $ps  # subsample

# spambase
./jobs/poison/primer.sh     $dsb $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dsb $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dsb $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dsb $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dsb $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dsb $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dsb $tt $mli $tf $us1 3  1440   $pl  # leaf_influence
./jobs/poison/primer.sh     $dsb $tt $mlo $tf $us1 5  1440   $ps  # loo
./jobs/poison/primer.sh     $dsb $tt $msb $tf $us1 5  1440   $ps  # subsample

# surgical
./jobs/poison/primer.sh     $dsg $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dsg $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dsg $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dsg $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dsg $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dsg $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dsg $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dsg $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dsg $tt $msb $tf $us1 5  1440  $ps  # subsample

# twitter
./jobs/poison/primer.sh     $dtw $tt $mr  $tf $us1 11  1440  $ps  # random
./jobs/poison/primer.sh     $dtw $tt $ms  $tf $us1 11  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dtw $tt $mis $tf $us1 11  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dtw $tt $mb2 $tf $us1 11  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dtw $tt $mls $tf $us0 11  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dtw $tt $mtx $tf $us1 28  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dtw $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dtw $tt $mlo $tf $us1 28  1440  $ps  # loo
./jobs/poison/primer.sh     $dtw $tt $msb $tf $us1 28  1440  $ps  # subsample

# vaccine
./jobs/poison/primer.sh     $dvc $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dvc $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dvc $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dvc $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dvc $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dvc $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dvc $tt $mli $tf $us1 3  10080 $p2  # leaf_influence
./jobs/poison/primer.sh     $dvc $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dvc $tt $msb $tf $us1 5  1440  $ps  # subsample

# wine
./jobs/poison/primer.sh     $dwn $tt $mr  $tf $us1 3  1440  $ps  # random
./jobs/poison/primer.sh     $dwn $tt $ms  $tf $us1 3  1440  $ps  # similarity2
./jobs/poison/primer.sh     $dwn $tt $mis $tf $us1 3  1440  $ps  # input_similarity
./jobs/poison/primer.sh     $dwn $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/poison/primer.sh     $dwn $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/poison/primer.sh     $dwn $tt $mtx $tf $us1 6  1440  $ps  # trex
# ./jobs/poison/primer.sh     $dwn $tt $mli $tf $us1 3  1440  $ps  # leaf_influence
./jobs/poison/primer.sh     $dwn $tt $mlo $tf $us1 5  1440  $ps  # loo
./jobs/poison/primer.sh     $dwn $tt $msb $tf $us1 5  1440  $ps  # subsample
