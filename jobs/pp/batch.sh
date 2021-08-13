tf=1.0

./jobs/pp/primer.sh 'adult' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'bank_marketing' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'bean' 'xgb' $tf 3 300 'short'

./jobs/pp/primer.sh 'casp' 'xgb' $tf 3 300 'short'

./jobs/pp/primer.sh 'credit_card' 'xgb' $tf 3 1440 'short'
./jobs/pp/primer.sh 'credit_card' 'svm' $tf 3 1440 'short'

./jobs/pp/primer.sh 'diabetes' 'xgb' $tf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'knn' $tf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'svm' 0.1 7 1440 'short'
./jobs/pp/primer.sh 'diabetes' 'mlp' $tf 7 600  'short'

./jobs/pp/primer.sh 'flight_delays' 'cb'   $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'xgb'  $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'skrf' $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'lr'   $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'dt'   $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'knn'  $tf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'svm'  0.1 7 1440 'short'
./jobs/pp/primer.sh 'flight_delays' 'mlp'  $tf 7 600  'short'

./jobs/pp/primer.sh 'msd' 'lgb'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'cb'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'xgb'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'skrf' $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'lr'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'dt'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'knn'  0.1  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'svm'  0.1  7 1440 'short'
./jobs/pp/primer.sh 'msd' 'mlp'  $tf  7 1440 'short'

./jobs/pp/primer.sh 'no_show' 'xgb' $tf  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'lr'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'dt'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'knn' 0.1  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'svm' 0.1  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'mlp' $tf  7 1440 'short'

./jobs/pp/primer.sh 'obesity' 'xgb' $tf  7 1440 'short'

./jobs/pp/primer.sh 'spambase' 'skrf' $tf 3 300 'short'

./jobs/pp/primer.sh 'surgical' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_binary' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_multiclass' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_regression' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'twitter' 'cb'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'xgb'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'skrf' $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'lr'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'dt'   $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'knn'  0.05 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'svm'  0.01 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'mlp'  $tf  7 1440 'short'

./jobs/pp/primer.sh 'vaccine' 'skrf' $tf 3 300 'short'
./jobs/pp/primer.sh 'vaccine' 'svm'  $tf 3 300 'short'
