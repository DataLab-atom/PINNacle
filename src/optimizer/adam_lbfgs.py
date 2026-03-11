import torch
from torch.optim import Optimizer
from src.optimizable.adam_lbfgs_optimizable import select_optimizer  # [OPTIMIZABLE]


class Adam_LBFGS(Optimizer):

    def __init__(
        self,
        params,
        switch_epoch=10000,
        adam_param={'lr': 1e-3, 'betas': (0.9, 0.999)},
        lbfgs_param={'lr': 1, 'max_iter': 20}
    ):
        self.params = list(params)
        self.switch_epoch = switch_epoch
        self.adam = torch.optim.Adam(self.params, **adam_param)
        self.lbfgs = torch.optim.LBFGS(self.params, **lbfgs_param)

        super().__init__(self.params, defaults={})

        self.state['current_step'] = 0
        self.state['loss_history'] = []   # 供 select_optimizer 使用

    def step(self, closure=None):
        self.state['current_step'] += 1

        # 计算当前梯度范数（供自适应切换策略使用）
        grad_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                grad_norm += p.grad.detach().norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        choice = select_optimizer(  # [OPTIMIZABLE]
            current_step=self.state['current_step'],
            switch_epoch=self.switch_epoch,
            loss_history=self.state['loss_history'],
            grad_norm=grad_norm,
        )

        if choice == 'adam':
            loss = self.adam.step(closure)
        else:
            if self.state['current_step'] == self.switch_epoch:
                print(f"Switch to LBFGS at epoch {self.switch_epoch}")
            loss = self.lbfgs.step(closure)

        # 记录 loss 历史（最多保留 100 步）
        if loss is not None:
            self.state['loss_history'].append(float(loss))
            if len(self.state['loss_history']) > 100:
                self.state['loss_history'].pop(0)

        return loss
