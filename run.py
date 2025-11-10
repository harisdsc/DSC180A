import subprocess
import sys

if __name__ == '__main__':
    subprocess.run(['git', 'pull'])

    args = sys.argv

    scripts = {'clean': 'src/preprocessing/clean.py',
               'ngrams': 'src/preprocessing/ngrams.py'}

    if len(args) > 1:
        script = args[1]
        if script == 'all':
            for key in scripts.keys():
                subprocess.run(['python3', scripts[key]])
        if script in scripts:
            subprocess.run(['python3', scripts[script]])
