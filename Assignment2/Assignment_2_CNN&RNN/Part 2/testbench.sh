mkdir outputs/
python cnn_train.py --optimizer=ADAM
python cnn_train.py --optimizer=SGD
python cnn_train.py --optimizer=ADAGRAD
python cnn_train.py --optimizer=ADAM --learning_rate=1e-2
python cnn_train.py --optimizer=ADAM --learning_rate=1e-6
