from invoke import task

# TODO:
# - list models
# - search models
# - display card
# - display help for model commands
# - run commands on models
# - initialize new model with template


@task
def clean(c, n : int):
    c.run(f"echo clean {n}")

@task
def build(c):
    c.run("echo build")
