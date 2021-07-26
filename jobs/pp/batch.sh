tf=1.0

./jobs/pp/primer.sh 'adult' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'bank_marketing' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'credit_card' 'svm' $tf 3 1440 'short'

./jobs/pp/primer.sh 'diabetes' 'knn' $tf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'svm' 0.1 7 1440 'short'
./jobs/pp/primer.sh 'diabetes' 'mlp' $tf 7 600  'short'

./jobs/pp/primer.sh 'flight_delays' 'knn' 7 $tf 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'svm' 7 0.1 1440 'short'
./jobs/pp/primer.sh 'flight_delays' 'mlp' 7 $tf 600  'short'

./jobs/pp/primer.sh 'no_show' 'lr'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'dt'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'knn' 0.1  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'svm' 0.1  7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'mlp' $tf  7 1440 'short'

./jobs/pp/primer.sh 'surgical' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_binary' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_multiclass' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'synth_regression' 'svm' $tf 3 300 'short'

./jobs/pp/primer.sh 'twitter' 'lr'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'dt'  $tf  7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'knn' 0.05 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'svm' 0.01 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'mlp' $tf  7 1440 'short'

./jobs/pp/primer.sh 'vaccine' 'svm' $tf 3 300 'short'
