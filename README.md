Tests where run with gas_rnn_alter, prediction plots and results where saved in plots/#.png and plots/#.txt

0-25: 3 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, patience=30, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
50-75: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, patience=30, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
100-125: 10 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, patience=30, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado


1000-1025: 5 past datapoints, 0.0075 L2 regularizer, layer size = 16, layers = GRU + Linear, patience=30, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
1050-1075: 5 past datapoints, 0.0025 L2 regularizer, layer size = 16, layers = GRU + Linear, patience=30, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado

1200-1225: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 200 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
1225-1250: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 100 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
1250-1275: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 500 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
1275-1300: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 1000 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado

1400-1425: 0.01 gaussian noise, 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 1000 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado
1425-1450: sin(x * pi) * 5 noise, 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, 1000 epochs, training with gas_terciero_caso_variavel and predicting gas_quinto_caso_alterado


2000-2030: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, [100,50,200] epochs, training with gas_primeiro_caso_variavel and predicting gas_quinto_caso_alterado
2000-2060: 5 past datapoints, 0.005 L2 regularizer, layer size = 16, layers = GRU + Linear, [100,50,200] epochs, training with gas_segundo_caso_variavel and predicting gas_quinto_caso_alterado


