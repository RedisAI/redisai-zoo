import redisai

# TODO: deploy TorchScript non-max suppression too


def deploy(model_file, device, key, tag='', host='127.0.0.1', port='6379'):

    r = redisai.Client(host=host, port=port)

    with open(model_file, 'rb') as f:
        model = f.read()

    r.modelset(key, 'TORCH', device, model, tag=tag)
