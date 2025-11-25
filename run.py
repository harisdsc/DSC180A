import subprocess
import time
import sys
import os

if __name__ == '__main__':
    print('Pulling latest from repo...')
    subprocess.run(['git', 'pull'])

    args = sys.argv

    scripts = {'clean': 'src/preprocessing/clean.py',
               'ngrams': 'src/preprocessing/ngrams.py',
               'train': 'src/models/train_model.py',
                'load': 'src/models/load_model.py'}

    configs = {'clean': 'configs/preprocessing/clean.json',
                'ngrams': 'configs/preprocessing/ngrams.json',
                'root': 'configs/models/root.json',
                'train': 'configs/models/train.json',
               'load': 'configs/models/load.json'}

    # Add current directory to PYTHONPATH so sub-scripts can import 'src'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')

    if len(args) > 1:
        script = args[1]
        config = args[2] if len(args) > 2 else None
        if script == 'all':
            for key in scripts.keys():
                subprocess.run(['python3', scripts[key]], env=env)
        if script in scripts:
            start_time = time.time()
            if config:
                subprocess.run(['python3', scripts[script], configs[config]], env=env)
            else:
                subprocess.run(['python3', scripts[script]], env=env)
            end_time = time.time()
            print(f"'{script}' completed in {end_time - start_time:.2f} seconds")