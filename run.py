import subprocess
import time
import os
import argparse

SCRIPTS = {
    'clean': 'src/preprocessing/clean.py',
    'ngrams': 'src/preprocessing/ngrams.py',
    'train': 'src/models/train_model.py',
    'load': 'src/models/load_model.py'
}

CONFIG_PATHS = {
    'clean': 'configs/preprocessing/clean.json',
    'ngrams': 'configs/preprocessing/ngrams.json',
    'catboost': 'configs/models/catboost.json',
    'logistic-regression': 'configs/models/logistic_regression.json',
    'transformer': 'configs/models/transformer.json'
}

def run_command(script_key, config_key=None):
    if script_key not in SCRIPTS:
        print(f"Error: Unknown script '{script_key}'. Available: {list(SCRIPTS.keys())}")
        return
    
    if config_key and config_key not in CONFIG_PATHS:
        print(f"Error: Unknown config '{config_key}'. Available: {list(CONFIG_PATHS.keys())}")
        return

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

    # Positional Arguments
    parser.add_argument('script')
    parser.add_argument('config', nargs='?')

    # Flags
    parser.add_argument('--no-pull', action='store_true')

    args = parser.parse_args()

    if not args.no_pull:
        print('Pulling latest from repo...')
        subprocess.run(['git', 'pull'], check=True)

    if args.script == 'all':
        for key in SCRIPTS.keys():
            run_command(key, args.config)
    else:
        run_command(args.script, args.config)