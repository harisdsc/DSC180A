## run.py
### Scripts
> `python3 run.py`**`<script>`**
- `ngrams` : Generate bi- and tri- gram frequencies of patterns in memo column then save counts to file.
  
- `clean` : Clean memos then save cleaned columns to file.
- `data` : Clean and apply feature engineering to data then save to file. 
---
> `python3 run.py`**`<script>`**`<model>`
- `train` : Train model on processed data then save to file.
- `load` : Load model from file, test accuracy, and print confusion matrix.
--- 
### Models 
> `python3 run.py <script>`**`<model>`**
- `catboost`
- `log-reg`
- `transformer`
---
- `all` : Run all scripts in sequential order (incl. train & load each model)
  - `python3 run.py all`
