import pip
import venv
import os

def setup_venv(c, venv_dir, requirements):
    venv.create(venv_dir, with_pip='True')
    pip = os.path.join(venv_dir, 'bin', 'pip')
    c.run(f'{pip} install -r {requirements}')
