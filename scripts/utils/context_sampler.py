import numpy as np
import torch as th

def get_L2_sphere(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size, device='cpu') * 2 - 1
        ctxs = ctxs / (th.sum((ctxs)**2, dim=-1).reshape(num, 1))**(1/2)
        ctxs = ctxs.to('cpu')
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
        ctxs = ctxs / (np.sum((ctxs)**2, axis=-1).reshape(num, 1))**(1/2)
    return ctxs

def get_unit_square(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size) * 2 - 1
    else:
        ctxs = np.random.rand(num, ctx_size) * 2 - 1
    return ctxs

def get_positive_square(ctx_size, num, torch=False):
    if torch:
        ctxs = th.rand(num, ctx_size)
    else:
        ctxs = np.random.rand(num, ctx_size)
    return ctxs

def get_categorical(ctx_size, num, torch=False):
    if torch:
        ctxs = th.zeros(num, ctx_size)
        ctxs[th.arange(num),th.randint(0, ctx_size, size=(num,))] = 1
    else:
        ctxs = np.zeros((num, ctx_size))
        ctxs[np.arange(num),np.random.randint(0, ctx_size, size=(num,))] = 1
    return ctxs

def get_natural_number(ctx_size, num, torch=False):
    '''
    Returns context vector of shape (num,1) with numbers in range [0, ctx_size]
    '''
    if torch:
        ctxs = th.randint(0, ctx_size, size=(num,1))
    else:
        ctxs = np.random.randint(0, ctx_size, size=(num,1))
    return ctxs

SAMPLERS = {"l2": get_L2_sphere, "unit_square": get_unit_square, "positive_square": get_positive_square, "categorical": get_categorical, "natural_numbers":get_natural_number}


'''
Transforms take a POSITIVE UNIT SQUARE context and transform it to the appropriate space
'''
def transform_L2_sphere(ctxs, torch=False, rescale=True):
    if torch:
        if rescale:
            ctxs = ctxs * 2 - 1
        if len(ctxs.shape) > 1:
            ctxs = ctxs / th.sum((ctxs)**2, dim=-1).reshape(ctxs.shape[0], 1)**(1/2)
        else:
            ctxs = ctxs / th.sum((ctxs)**2, dim=-1)**(1/2)
    else:
        if rescale:
            ctxs = ctxs * 2 - 1
        if len(ctxs.shape) > 1:
            ctxs = ctxs / np.sum((ctxs)**2, axis=-1).reshape(ctxs.shape[0], 1)**(1/2)
        else:
            ctxs = ctxs / np.sum((ctxs)**2, axis=-1)**(1/2)
    return ctxs

def transform_unit_square(ctxs, torch=False):
    if torch:
        ctxs = ctxs * 2 - 1
    else:
        ctxs = ctxs * 2 - 1
    return ctxs

def transform_positive_square(ctxs, torch=False):
    '''
    identity
    '''
    return ctxs

def transform_categorical(ctxs, torch=False):
    '''
    assumes ctxs is already categorical
    '''
    return ctxs

TRANSFORMS = {"l2": transform_L2_sphere, "unit_square": transform_unit_square, "positive_square": transform_positive_square, "categorical": transform_categorical}


if __name__ == "__main__":
    # print("L2:")
    # print(get_L2_sphere(3, 5))
    # print(get_L2_sphere(3, 5, True)) 

    # print("unit square:")
    # print(get_unit_square(3, 5))
    # print(get_unit_square(3, 5, True)) 

    # print("positive square:")
    # print(get_positive_square(3, 5))
    # print(get_positive_square(3, 5, True))

    # print("categorical:")
    # print(get_categorical(3, 5))
    # print(get_categorical(3, 5, True)) 

    print(transform_L2_sphere(np.array([1.2, -1.05]), rescale=False))
    print(transform_L2_sphere(np.array([[1.2, -1.05]]), rescale=False))

    # print(transform_L2_sphere(th.rand(3), torch=True))

    pass
