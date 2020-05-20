from invoke import task
import shutil
import os
import util
import json
import redisai

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, 'build')

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6379


def source_dir(zooconf):
    return os.path.dirname(os.path.abspath(zooconf))


def build_dir(zooconf):
    return os.path.join(BUILD_DIR, os.path.relpath(source_dir(zooconf), ROOT_DIR))


def export_dir(zooconf):
    return os.path.join(source_dir(zooconf), 'export')


def predict_dir(zooconf):
    return os.path.join(source_dir(zooconf), 'predict')


def asset_dir(zooconf):
    return os.path.join(source_dir(zooconf), 'asset')


def predict_env_dir(zooconf):
    return os.path.join(build_dir(zooconf), 'venv')


def interp_path(zooconf):
    return os.path.join(predict_env_dir(zooconf), 'bin', 'python')


def search_models(query=None):
    from glob import glob
    pattern = "zooconf.json"

    out = []
    for d, _, _ in os.walk(ROOT_DIR):
        matches = glob(os.path.join(d, pattern))
        if matches and len(matches) == 1:
            zooconf = matches[0]
            with open(zooconf, 'r') as f:
                config = json.load(f)
            found = True
            if query:
                found = query in config['name'] or \
                        query in config['framework'] or \
                        any(query in el for el in config['tags'])
            if found:
                out.append([config['name'], zooconf])
    return out


def search_zooconf(query):
    matches = search_models(query)

    zooconf = None

    if matches and len(matches) == 1:
        _, zooconf = matches[0]

    return zooconf


@task
def clean(c,
          name='',
          zooconf='',
          all=False):

    if all:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    if not zooconf:
        zooconf = search_zooconf(name)

    if not zooconf:
        print(f'No entry found in zoo for {name}')
        return

    shutil.rmtree(build_dir(zooconf), ignore_errors=True)


@task
def export(c,
           name='',
           zooconf='',
           arg=[],
           args=False):

    if not zooconf:
        zooconf = search_zooconf(name)

    if not zooconf:
        print(f'No entry found in zoo for {name}')
        return

    with open(zooconf, 'r') as f:
        config = json.load(f)['export']

    if args:
        for k in config['args']:
            print(k)
        return

    # all_args = {k: v['value'] for k, v in config['args'].items()}
    all_args = config['args']
    cli_args = dict(el.split('=') for el in arg)
    all_args.update(cli_args)

    args_string = ' '.join(f'--build-arg {k}={v}' for k, v in all_args.items())
    print(args_string)

    c.run(f'DOCKER_BUILDKIT=1 \
            docker build --target=export {args_string}\
                        --output type=local,dest={build_dir(zooconf)} \
                        {export_dir(zooconf)}/')


@task  # (export)
def deploy(c,
           name='',
           zooconf='',
           prefix='',
           tag='',
           device='default',
           host=DEFAULT_HOST,
           port=DEFAULT_PORT):

    if not zooconf:
        zooconf = search_zooconf(name)

    if not zooconf:
        print('No entry found in zoo for {name}')
        return

    with open(zooconf, 'r') as f:
        config = json.load(f)['deploy']

    r = redisai.Client(host=host, port=port)

    for el in config:
        if el['type'] == 'model':
            model_path = os.path.join(build_dir(zooconf), el['filename'])
            with open(model_path, 'rb') as f:
                model = f.read()
            device = el['device'] if device == 'default' else device
            r.modelset(prefix+el['key'], el['backend'], device, model,
                       inputs=el.get('inputs'), outputs=el.get('outputs'),
                       tag=el.get('tag'))

        elif el['type'] == 'script':
            script_path = os.path.join(asset_dir(zooconf), el['filename'])
            with open(script_path, 'rb') as f:
                script = f.read()
            device = el['device'] if device == 'default' else device
            r.scriptset(prefix+el['key'], device, script, tag=el.get('tag'))


@task
def predict(c,
            name='',
            zooconf='',
            arg=[],
            args=False,
            prefix='',
            language='python',
            docker=False,
            host=DEFAULT_HOST,
            port=DEFAULT_PORT):

    if not zooconf:
        zooconf = search_zooconf(name)

    if not zooconf:
        print(f'No entry found in zoo for {name}')
        return

    with open(zooconf, 'r') as f:
        config = json.load(f)['predict']

    if args:
        for k in config[language]['args']:
            print(k)
        return

    if language == 'python' and not docker:
        interp = interp_path(zooconf)
        predict_python_dir = os.path.join(predict_dir(zooconf), 'python')
        predict_script = os.path.join(predict_python_dir, config['python']['script'])

        if not os.path.isfile(interp):
            util.setup_venv(c, predict_env_dir(zooconf), os.path.join(predict_python_dir, 'requirements.txt'))

        # all_args = {k: v['value'] for k, v in config['python']['args'].items()}
        all_args = config['python']['args']
        cli_args = dict(el.split('=') for el in arg)
        all_args.update(cli_args)

        args_string = ' '.join(f'--{k}={v}' for k, v in all_args.items())

        c.run(f'{interp} {predict_script} --prefix={prefix} --host={host} --port={port} {args_string}')

    # elif docker:
    #     build_args_string = ' '.join(f'--build-arg {k}={v}' for k, v in config["args"].items())

    #     c.run(f'DOCKER_BUILDKIT=1 \
    #             docker build --target=predict \
    #                         --build-arg {build_args_string} \
    #                         predict/python/')


@task(name="list")
def list_models(c):
    matches = search_models(query=None)

    for name, zooconf in matches:
        print(f"{name}\t\t{os.path.relpath(zooconf, ROOT_DIR)}")


@task(name="find")
def query_models(c,
                 query):
    matches = search_models(query)

    for name, zooconf in matches:
        print(f"{name}\t\t{os.path.relpath(zooconf, ROOT_DIR)}")
