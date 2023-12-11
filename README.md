# Stock price forecasting in Indonesia stock exchange using deep learning: a comparative study
In 2022, the Indonesia Stock Exchange (IDX) listed 825 companies, making it  challenging  to  identify  low-risk  companies.  Stock  price  forecasting  and price movement prediction are vital issues in financial works. Deep learning has previously been implemented for stock market analysis, with promising results.  Because  of  the  differences  in  architecture  and  stock issuers  in  each study report, a consensus on the best stock price forecasting model has yet to be  reached.  We  present  a  methodology  for  comparing  the  performance  of convolutional  neural  networks  (CNN),  gated  recurrent  units  (GRU),  long short-term  memory  (LSTM),  and  graph  convolutional  networks  (GCN) layers.  The  four layers  type  combination  yields  11  architectures  with  two layers stacked maximum, and the architectures are performance compared in stock price  predicting.  The dataset  consists of open, highest,  lowest,  closed price, and volume transactions and has 2,588,451 rows from 727 companies in  IDX.  The  best  performance  architecture  was  chosen  by  a  vote  based  on the coefficient of determination (R2), mean squared error (MSE), root mean square  error  (RMSE),  mean  absolute  percent  error  (MAPE),  and  f1-score. TFGRU   is   the   best   architecture,   producing   the   finest   results   on   315 companies with an average score of RMSE is 553.327, MAPE is 0.858, and f1-score is 0.456.

DOI: https://doi.org/10.11591/ijece.v14i1.pp861-869

# How to run
```
python.exe .\forecasting.py --code BBCA --type 1 --lookback '5,10,15,20,25' --scaler standard --batch_size 128 --epoch 150 --callbacks 1 --model TFGRU 
```
I'd like you to please open the file forecasting.py for more details. Existing models are CNN, CNN-GRU, CNN-LSTM, GCN-GRU, GCN-LSTM, GRU, GRU-CNN, GRU-LSTM, LSTM, LSTM-CNN, LSTM-GRU.

# Custom architecture
```
import tensorflow as tf

class TFGRU:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def build_model(self,
                    input_shape,
                    dropout=0.25,
                    unit=128
                    ):
        self.dropout = dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        x = tf.keras.layers.GRU(unit, activation="tanh")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # output layer
        outputs = tf.keras.layers.Dense(self.n_classes)(x)

        return tf.keras.Model(inputs, outputs)
```

Change the layers with a custom layer with accept keras input (x), put custom architecture in folder .\models, and import custom architecture in file .\models\__init__.py
