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
    'root': 'configs/models/root.json',
    'train': 'configs/models/train.json',
    'load': 'configs/models/load.json',
    'catboost': 'configs/models/catboost.json' 
}

MODELS_PATHS = {
    'catboost' : 'src/models/catboost.cbm'
}

def run_command(script_key, config_key=None, model_key=None):
    if script_key not in SCRIPTS:
        print(f"Error: Unknown script '{script_key}'. Available: {list(SCRIPTS.keys())}")
        return

    script_path = SCRIPTS[script_key]
    
    cmd = ['python3', script_path]

    if config_key:
        config_path = CONFIG_PATHS.get(config_key, config_key)
        cmd.append(config_path)
    
    if model_key:
        model_path = MODELS_PATHS.get(model_key, model_key)
        cmd.append(model_path)

    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_key}: {e}")
        
    print(f"'{script_key}' completed in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Positional Arguments
    parser.add_argument('script')
    parser.add_argument('config', nargs='?')
    parser.add_argument('model', nargs='?')

    # Flags
    parser.add_argument('--no-pull', action='store_true')

    args = parser.parse_args()

    if not args.no_pull:
        print('Pulling latest from repo...')
        # Added check=True to catch git errors
        try:
            subprocess.run(['git', 'pull'], check=True)
        except subprocess.CalledProcessError:
            print("Git pull failed, continuing execution...")

    if args.script == 'all':
        for key in SCRIPTS.keys():
            run_command(key, args.config, args.model)
    else:
        run_command(args.script, args.config, args.model)