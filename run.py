import subprocess
import time
import sys

if __name__ == '__main__':
    subprocess.run(['git', 'pull'])

    args = sys.argv

    scripts = {'clean': 'src/preprocessing/clean.py',
               'train': 'src/models/train.py',
               'ngrams': 'src/preprocessing/ngrams.py'}

    configs = {'clean': 'configs/preprocessing/clean.json',
               'root': 'configs/preprocessing/root.json',
               'ngrams': 'configs/ngrams.json'}

    if len(args) > 1:
        script = args[1]
        config = args[2] if len(args) > 2 else None
        if script == 'all':
            for key in scripts.keys():
                subprocess.run(['python3', scripts[key]])
        if script in scripts:
            start_time = time.time()
            if config:
                subprocess.run(['python3', scripts[script], configs[config]])
            else:
                subprocess.run(['python3', scripts[script]])
            end_time = time.time()
            print(f"{script} completed in {end_time - start_time:.2f} seconds")
