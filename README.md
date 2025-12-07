## Assignment Structure
```
finm32500-assignment11/
│
├── Data/
│   ├── market_data_ml.csv        # OHLCV price data
│   └── tickers-1.csv             # List of tickers
│
├── plots/                    
│   ├── confusion_*.png
│   ├── equity_*.png
│   ├── residuals_*.png
│   ├── pred_dist_*.png
│   └── feature_importance_*.png
│
├── feature_engineering.py        # Builds predictive features + labels
├── train_model.py                # Trains models, evaluates, backtests
├── backtest.py                   # PnL & equity curve simulation
├── signal_generator.py           # Converts predictions to signals
├── comparison.md                 # Generated performance report
├── main.py                       # Runs full ML and Backtest pipeline
│
├── features_config.json          # Defines which engineered features to train on
├── model_params.json             # Hyperparameters for each ML model
│
├── tests/                
│   ├── feature_engineering_test.py
│   ├── train_model_test.py
│   ├── backtest_test.py
│   └── signal_generator_test.py
│
├── requirements.txt     
└── README.md                    
```


## Run Full Analysis
```
python main.py
```
## Run all tests
```
pytest tests
```