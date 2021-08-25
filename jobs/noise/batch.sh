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
ms='similarity'
ml='loss'
mtg='target'
msb='subsample'

ps='short'
pl='long'

tf=0.25  # trunc_frac

us0=0  # leaf_influence update set
us1=-1

st1='self'
st2='test_sum' # strategy

# adult
./jobs/noise/primer.sh $dad $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dad $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dad $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dad $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dad $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dad $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dad $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dad $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# bank_marketing
./jobs/noise/primer.sh $dbm $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dbm $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dbm $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dbm $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dbm $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dbm $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dbm $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dbm $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# bean
./jobs/noise/primer.sh $dbn $tt $mr  $tf $us1 $st2 3  600  $ps  # random
./jobs/noise/primer.sh $dbn $tt $ml  $tf $us1 $st1 3  600  $ps  # loss
./jobs/noise/primer.sh $dbn $tt $ms  $tf $us1 $st2 3  600  $ps  # similarity
./jobs/noise/primer.sh $dbn $tt $mb2 $tf $us1 $st2 3  600  $ps  # boostin2
./jobs/noise/primer.sh $dbn $tt $mls $tf $us0 $st2 3  600  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dbn $tt $mtx $tf $us1 $st2 6  600  $ps  # trex
./jobs/noise/primer.sh $dbn $tt $mlo $tf $us1 $st2 5  600  $ps  # loo
./jobs/noise/primer.sh $dbn $tt $msb $tf $us1 $st2 5  600  $ps  # subsample

# compas
./jobs/noise/primer.sh $dco $tt $mr  $tf $us1 $st2 3  600  $ps  # random
./jobs/noise/primer.sh $dco $tt $ml  $tf $us1 $st1 3  600  $ps  # loss
./jobs/noise/primer.sh $dco $tt $ms  $tf $us1 $st2 3  600  $ps  # similarity
./jobs/noise/primer.sh $dco $tt $mb2 $tf $us1 $st2 3  600  $ps  # boostin2
./jobs/noise/primer.sh $dco $tt $mls $tf $us0 $st2 3  600  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dco $tt $mtx $tf $us1 $st2 6  600  $ps  # trex
./jobs/noise/primer.sh $dco $tt $mlo $tf $us1 $st2 5  600  $ps  # loo
./jobs/noise/primer.sh $dco $tt $msb $tf $us1 $st2 5  600  $ps  # subsample

# concrete
./jobs/noise/primer.sh $dcn $tt $mr  $tf $us1 $st2 3  600  $ps  # random
./jobs/noise/primer.sh $dcn $tt $ml  $tf $us1 $st1 3  600  $ps  # loss
./jobs/noise/primer.sh $dcn $tt $ms  $tf $us1 $st2 3  600  $ps  # similarity
./jobs/noise/primer.sh $dcn $tt $mb2 $tf $us1 $st2 3  600  $ps  # boostin2
./jobs/noise/primer.sh $dcn $tt $mls $tf $us0 $st2 3  600  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dcn $tt $mtx $tf $us1 $st2 6  600  $ps  # trex
./jobs/noise/primer.sh $dcn $tt $mlo $tf $us1 $st2 5  600  $ps  # loo
./jobs/noise/primer.sh $dcn $tt $msb $tf $us1 $st2 5  600  $ps  # subsample

# credit_card
./jobs/noise/primer.sh $dcc $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dcc $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dcc $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dcc $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dcc $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dcc $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dcc $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dcc $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# diabetes
./jobs/noise/primer.sh $ddb $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $ddb $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $ddb $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $ddb $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $ddb $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $ddb $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $ddb $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $ddb $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# energy
./jobs/noise/primer.sh $den $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $den $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $den $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $den $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $den $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $den $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $den $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $den $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# flight_delays
./jobs/noise/primer.sh $dfd $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dfd $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dfd $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dfd $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dfd $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dfd $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dfd $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dfd $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# german_credit
./jobs/noise/primer.sh $dgc $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dgc $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dgc $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dgc $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dgc $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dgc $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dgc $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dgc $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# htru2
./jobs/noise/primer.sh $dht $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dht $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dht $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dht $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dht $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dht $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dht $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dht $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# life
./jobs/noise/primer.sh $dlf $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dlf $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dlf $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dlf $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dlf $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dlf $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dlf $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dlf $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# msd
./jobs/noise/primer.sh $dms $tt $mr  $tf $us1 $st2 20  1440  $ps  # random
./jobs/noise/primer.sh $dms $tt $ml  $tf $us1 $st1 20  1440  $ps  # loss
./jobs/noise/primer.sh $dms $tt $ms  $tf $us1 $st2 20  1440  $ps  # similarity
./jobs/noise/primer.sh $dms $tt $mb2 $tf $us1 $st2 20  1440  $ps  # boostin2
./jobs/noise/primer.sh $dms $tt $mls $tf $us0 $st2 20  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dms $tt $mtx $tf $us1 $st2 20  1440  $ps  # trex
./jobs/noise/primer.sh $dms $tt $mlo $tf $us1 $st2 20  1440  $ps  # loo
./jobs/noise/primer.sh $dms $tt $msb $tf $us1 $st2 20  1440  $ps  # subsample

# naval
./jobs/noise/primer.sh $dnv $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dnv $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dnv $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dnv $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dnv $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dnv $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dnv $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dnv $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# no_show
./jobs/noise/primer.sh $dns $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dns $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dns $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dns $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dns $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dns $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dns $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dns $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# obesity
./jobs/noise/primer.sh $dob $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dob $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dob $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dob $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dob $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dob $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dob $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dob $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# power
./jobs/noise/primer.sh $dpw $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dpw $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dpw $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dpw $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dpw $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dpw $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dpw $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dpw $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# protein
./jobs/noise/primer.sh $dpr $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dpr $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dpr $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dpr $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dpr $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dpr $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dpr $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dpr $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# spambase
./jobs/noise/primer.sh $dsb $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dsb $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dsb $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dsb $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dsb $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dsb $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dsb $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dsb $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# surgical
./jobs/noise/primer.sh $dsg $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dsg $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dsg $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dsg $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dsg $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dsg $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dsg $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dsg $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# twitter
./jobs/noise/primer.sh $dtw $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dtw $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dtw $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dtw $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dtw $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dtw $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dtw $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dtw $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# vaccine
./jobs/noise/primer.sh $dvc $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dvc $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dvc $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dvc $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dvc $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dvc $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dvc $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dvc $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample

# wine
./jobs/noise/primer.sh $dwn $tt $mr  $tf $us1 $st2 3  1440  $ps  # random
./jobs/noise/primer.sh $dwn $tt $ml  $tf $us1 $st1 3  1440  $ps  # loss
./jobs/noise/primer.sh $dwn $tt $ms  $tf $us1 $st2 3  1440  $ps  # similarity
./jobs/noise/primer.sh $dwn $tt $mb2 $tf $us1 $st2 3  1440  $ps  # boostin2
./jobs/noise/primer.sh $dwn $tt $mls $tf $us0 $st2 3  1440  $ps  # leaf_influenceSP
./jobs/noise/primer.sh $dwn $tt $mtx $tf $us1 $st2 6  1440  $ps  # trex
./jobs/noise/primer.sh $dwn $tt $mlo $tf $us1 $st2 5  1440  $ps  # loo
./jobs/noise/primer.sh $dwn $tt $msb $tf $us1 $st2 5  1440  $ps  # subsample
