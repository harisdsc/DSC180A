import subprocess
import time
import os
import argparse

SCRIPTS = {
    'ngrams': 'src/preprocessing/ngrams.py',
    'clean': 'src/preprocessing/clean.py',
    'data': 'src/models/load_data.py',
    'train': 'src/models/train_model.py',
    'load': 'src/models/load_model.py'
}

CONFIG_PATHS = {
    'clean': 'configs/preprocessing/clean.json',
    'ngrams': 'configs/preprocessing/ngrams.json',
    'catboost': 'configs/models/catboost.json',
    'log-reg': 'configs/models/logistic_regression.json',
    'transformer': 'configs/models/transformer.json'
}

def run_command(script_key, config_key=None):
    script_path = SCRIPTS[script_key]
    
    cmd = ['python3', script_path]
    cmd_string = [script_path.split('/')[-1]]

    if config_key:
        config_path = CONFIG_PATHS.get(config_key, config_key)
        cmd.append(config_path)
        cmd_string.append(config_path.split('/')[-1])

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')

    print(f"Running: {', '.join(cmd_string)}")
    
    start_time = time.time()
    subprocess.run(cmd, env=env, check=True)
    print(f"'{script_key}{' ' + config_key if config_key else ''}' completed in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('config', nargs='?')

    args = parser.parse_args()

    print('Pulling latest from repo...')
    subprocess.run(['git', 'pull'], check=True)

    if args.script == 'all':
        run_command('ngrams', 'ngrams')
        run_command('clean', 'clean')
        run_command('data')

        for model in ['catboost', 'log-reg', 'transformer']:
            run_command('train', model)
            run_command('load', model)
    else:
        run_command(args.script, args.config)