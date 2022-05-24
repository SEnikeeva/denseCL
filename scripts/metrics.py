from ignite.metrics import ConfusionMatrix, mIoU
from ignite.engine import Engine


def accuracy(output, target, num_classes=19):
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    cm = ConfusionMatrix(num_classes=num_classes)
    metric = mIoU(cm, ignore_index=0)
    metric.attach(default_evaluator, 'miou')
    state = default_evaluator.run([[output, target]])
    return state.metrics['miou']