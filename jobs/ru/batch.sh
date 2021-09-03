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
ml='loss'
mtg='target'
msb='subsample'

ps='short'
pl='long'

tf=0.25  # trunc_frac

us0=0  # leaf_influence update set
us1=-1

# adult
./jobs/ru/primer.sh $dad $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dad $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dad $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dad $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dad $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dad $tt $msb $tf $us1 3  1440  $ps  # subsample

# bank_marketing
./jobs/ru/primer.sh $dbm $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dbm $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dbm $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dbm $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dbm $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dbm $tt $msb $tf $us1 3  1440  $ps  # subsample

# bean
./jobs/ru/primer.sh $dbn $tt $ms  $tf $us1 3  600  $ps  # similarity
./jobs/ru/primer.sh $dbn $tt $mb2 $tf $us1 3  600  $ps  # boostin2
./jobs/ru/primer.sh $dbn $tt $mls $tf $us0 3  600  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dbn $tt $mtx $tf $us1 6  600  $ps  # trex
./jobs/ru/primer.sh $dbn $tt $mlo $tf $us1 3  600  $ps  # loo
./jobs/ru/primer.sh $dbn $tt $msb $tf $us1 3  600  $ps  # subsample

# compas
./jobs/ru/primer.sh $dco $tt $ms  $tf $us1 3  600  $ps  # similarity
./jobs/ru/primer.sh $dco $tt $mb2 $tf $us1 3  600  $ps  # boostin2
./jobs/ru/primer.sh $dco $tt $mls $tf $us0 3  600  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dco $tt $mtx $tf $us1 6  600  $ps  # trex
./jobs/ru/primer.sh $dco $tt $mlo $tf $us1 3  600  $ps  # loo
./jobs/ru/primer.sh $dco $tt $msb $tf $us1 3  600  $ps  # subsample

# concrete
./jobs/ru/primer.sh $dcn $tt $ms  $tf $us1 3  600  $ps  # similarity
./jobs/ru/primer.sh $dcn $tt $mb2 $tf $us1 3  600  $ps  # boostin2
./jobs/ru/primer.sh $dcn $tt $mls $tf $us0 3  600  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dcn $tt $mtx $tf $us1 6  600  $ps  # trex
./jobs/ru/primer.sh $dcn $tt $mlo $tf $us1 3  600  $ps  # loo
./jobs/ru/primer.sh $dcn $tt $msb $tf $us1 3  600  $ps  # subsample

# credit_card
./jobs/ru/primer.sh $dcc $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dcc $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dcc $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dcc $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dcc $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dcc $tt $msb $tf $us1 3  1440  $ps  # subsample

# diabetes
./jobs/ru/primer.sh  $ddb $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $ddb $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $ddb $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh  $ddb $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer2.sh $ddb $tt $mlo $tf $us1 3  2160  $pl  # loo
./jobs/ru/primer.sh  $ddb $tt $msb $tf $us1 3  1440  $ps  # subsample

# energy
./jobs/ru/primer.sh $den $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $den $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $den $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $den $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $den $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $den $tt $msb $tf $us1 3  1440  $ps  # subsample

# flight_delays
./jobs/ru/primer.sh  $dfd $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $dfd $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dfd $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh  $dfd $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer2.sh $dfd $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh  $dfd $tt $msb $tf $us1 3  1440  $ps  # subsample

# german_credit
./jobs/ru/primer.sh $dgc $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dgc $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dgc $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dgc $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dgc $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dgc $tt $msb $tf $us1 3  1440  $ps  # subsample

# htru2
./jobs/ru/primer.sh $dht $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dht $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dht $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dht $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dht $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dht $tt $msb $tf $us1 3  1440  $ps  # subsample

# life
./jobs/ru/primer.sh $dlf $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dlf $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dlf $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dlf $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dlf $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dlf $tt $msb $tf $us1 3  1440  $ps  # subsample

# msd
./jobs/ru/primer.sh  $dms $tt $ms  $tf $us1 20  1440  $ps  # similarity
./jobs/ru/primer.sh  $dms $tt $mb2 $tf $us1 20  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dms $tt $mls $tf $us0 20  1440  $ps  # leaf_influenceSP
./jobs/ru/primer2.sh $dms $tt $mtx $tf $us1 28  1440  $ps  # trex
./jobs/ru/primer.sh  $dms $tt $mlo $tf $us1 20  1440  $ps  # loo
./jobs/ru/primer2.sh $dms $tt $msb $tf $us1 20  1440  $ps  # subsample

# naval
./jobs/ru/primer.sh $dnv $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dnv $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dnv $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dnv $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dnv $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dnv $tt $msb $tf $us1 3  1440  $ps  # subsample

# no_show
./jobs/ru/primer.sh  $dns $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $dns $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dns $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh  $dns $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer2.sh $dns $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh  $dns $tt $msb $tf $us1 3  1440  $ps  # subsample

# obesity
./jobs/ru/primer.sh  $dob $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $dob $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dob $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh  $dob $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer2.sh $dob $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh  $dob $tt $msb $tf $us1 3  1440  $ps  # subsample

# power
./jobs/ru/primer.sh $dpw $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dpw $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dpw $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dpw $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dpw $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dpw $tt $msb $tf $us1 3  1440  $ps  # subsample

# protein
./jobs/ru/primer.sh  $dpr $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $dpr $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dpr $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh  $dpr $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer2.sh $dpr $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh  $dpr $tt $msb $tf $us1 3  1440  $ps  # subsample

# spambase
./jobs/ru/primer.sh $dsb $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dsb $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dsb $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dsb $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dsb $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dsb $tt $msb $tf $us1 3  1440  $ps  # subsample

# surgical
./jobs/ru/primer.sh $dsg $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dsg $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dsg $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dsg $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dsg $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dsg $tt $msb $tf $us1 3  1440  $ps  # subsample

# twitter
./jobs/ru/primer.sh  $dtw $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh  $dtw $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh  $dtw $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer2.sh $dtw $tt $mtx $tf $us1 28 1440  $ps  # trex
./jobs/ru/primer.sh  $dtw $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer2.sh $dtw $tt $msb $tf $us1 3  1440  $ps  # subsample

# vaccine
./jobs/ru/primer.sh $dvc $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dvc $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dvc $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dvc $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dvc $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dvc $tt $msb $tf $us1 3  1440  $ps  # subsample

# wine
./jobs/ru/primer.sh $dwn $tt $ms  $tf $us1 3  1440  $ps  # similarity
./jobs/ru/primer.sh $dwn $tt $mb2 $tf $us1 3  1440  $ps  # boostin2
./jobs/ru/primer.sh $dwn $tt $mls $tf $us0 3  1440  $ps  # leaf_influenceSP
./jobs/ru/primer.sh $dwn $tt $mtx $tf $us1 6  1440  $ps  # trex
./jobs/ru/primer.sh $dwn $tt $mlo $tf $us1 3  1440  $ps  # loo
./jobs/ru/primer.sh $dwn $tt $msb $tf $us1 3  1440  $ps  # subsample
