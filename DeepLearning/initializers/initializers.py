import re
from ast import literal_eval as _eval

from DeepLearning.activations.activations import Identity, ActivationBase, Sigmoid, ReLU, Affine, LeakyReLU
from DeepLearning.optimizers.optimizers import SGD, OptimizerBase
from DeepLearning.schedulers.schedulers import SchedulerBase, ConstantScheduler


class ActivationInitializer(object):
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            act = Identity()
        elif isinstance(param, ActivationBase):
            act = param
        elif isinstance(param, str):
            act = self.init_from_str(param)
        else:
            raise ValueError("Unknown activation: {}".format(param))
        return act

    def init_from_str(self, param):
        act_str = param.lower()
        if act_str == "relu":
            act_fn = ReLU()
        elif act_str == "sigmoid":
            act_fn = Sigmoid()
        elif act_str == "identity":
            act_fn = Identity()
        elif "affine" in act_str:
            r = r"affine\(slope=(.*), intercept=(.*)\)"
            slope, intercept = re.match(r, act_str).groups()
            act_fn = Affine(float(slope), float(intercept))
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            act_fn = LeakyReLU(float(alpha))
        else:
            raise ValueError("Unknown activation: {}".format(act_str))
        return act_fn


class SchedulerInitializer(object):
    def __init__(self, lr=None, param=None):
        if all([lr is None, param is None]):
            raise ValueError("lr and param cannot both be 'None'")
        self.lr = lr
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            scheduler = ConstantScheduler(self.lr)
        elif isinstance(param, SchedulerBase):
            scheduler = param
        elif isinstance(param, str):
            scheduler = self.init_from_str()
        elif isinstance(param, dict):
            scheduler = self.init_from_dict()
        else:
            raise ValueError("Unknown scheduler: {}".format(param))
        return scheduler

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        sch_str = self.param.lower()
        kwargs = {i: _eval(j) for i, j in re.findall(r, sch_str)}

        if "constant" in sch_str:
            scheduler = ConstantScheduler(**kwargs)
        else:
            raise NotImplementedError("{}".format(sch_str))
        return scheduler

    def init_from_dict(self):
        S = self.param
        sc = S["hyperparameters"] if ["hyperparameters"] in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        if sc["id"] == "ConstantScheduler":
            scheduler = ConstantScheduler()
        else:
            raise NotImplementedError("{}".format(sc['id']))
        scheduler.set_param(sc)
        return scheduler


class OptimizerInitializer(object):
    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str()
        elif isinstance(param, dict):
            opt = self.init_from_dict()
        else:
            raise ValueError("{} is not exist".format(param))
        return opt

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = self.param.lower()
        kwargs = {i: _eval(j) for i, j in re.findall(r, opt_str)}
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        else:
            raise NotImplementedError("{} is not exist".format(opt_str))
        return optimizer

    def init_from_dict(self):
        D = self.param
        cc = D["cache"] if "cache" in D else None
        op = D["hyperparameters"] if "hyperparameters" in D else None

        if op is None:
            raise ValueError("`param` dictionary has no `hyperparemeters` key")

        if op["id"] == "SGD":
            optimizer = SGD()
        else:
            raise NotImplementedError("{}".format(op["id"]))
        optimizer.set_params(op, cc)
        return optimizer
