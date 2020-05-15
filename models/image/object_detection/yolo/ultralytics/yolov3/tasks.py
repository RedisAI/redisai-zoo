from invoke import task
import shutil
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, 'build')
PREDICT_ENV_DIR = os.path.join(ROOT_DIR, 'predict', 'env')
MODEL_FILE = 'model.pt'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6379


@task
def clean(c):
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.rmtree(PREDICT_ENV_DIR, ignore_errors=True)


@task
def export(c):
    if os.path.isfile(os.path.join(BUILD_DIR, MODEL_FILE)):
        return
    c.run(f'DOCKER_BUILDKIT=1 \
            docker build --target=export \
                        --build-arg MODEL_FILE={MODEL_FILE} \
                        --output type=local,dest={BUILD_DIR} \
                        export/')


@task(export)
def deploy(c,
           device='cpu',
           key='yolov3',
           tag='',
           host=DEFAULT_HOST,
           port=DEFAULT_PORT):
    from deploy.deploy import deploy
    deploy(model_file=os.path.join(BUILD_DIR, MODEL_FILE),
           device=device,
           key=key,
           tag=tag,
           host=host,
           port=port)



def setup_venv(c, venv_dir, requirements):
    import venv
    venv.create(venv_dir, with_pip='True')
    pip = os.path.join(venv_dir, 'bin', 'pip')
    c.run(f'{pip} install -r {requirements}')


@task
def predict(c,
            filenames='',
            key='yolov3',
            host=DEFAULT_HOST,
            port=DEFAULT_PORT):

    interp = os.path.join(PREDICT_ENV_DIR, 'bin', 'python')
    predict = os.path.join('predict', 'predict.py')

    if not os.path.isdir(venv_dir):
        setup_venv(c, PREDICT_ENV_DIR, os.path.join('predict', 'requirements.txt'))

    c.run(f'{interp} {predict} --key={key} --host={host} --port={port} --filenames={filenames}')
