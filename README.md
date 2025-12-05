## run.py
---
### Scripts
> `python3 run.py`**`<script>`**

- `all` : Runs all scripts in sequential order.

- `ngrams` : Generate bi- and tri- gram frequencies of patterns in memo column then save counts to file.
  
- `clean` : Clean memo column then save cleaned memos to file.
---

- `train` : Train model and save to file.
  - `python3 run.py train <model>`
- `load` : Load model from file, test accuracy, and print confusion matrix.
  - `python3 run.py load <model>`
--- 
### Models 
> `python3 run.py <script>`**`<model>`**

- `catboost`
- `logistic-regression`
- `transformer`