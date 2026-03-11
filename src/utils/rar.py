import numpy as np

from src.optimizable.sampling_optimizable import select_resample_points


def rar_wrapper(pde, model, conf):
    data = model.data
    train = model.train

    def wrapper(*args, **kwargs):
        total_iter = kwargs['iterations']
        interval, count = conf['interval'], conf['count']

        assert total_iter % interval == 0
        kwargs['iterations'] = interval

        for i in range(total_iter // interval):
            if i == 0:
                train(*args, **kwargs)
                continue

            X = model.train_state.X_train
            f = model.predict(X, operator=pde.pde)
            err = np.abs(f).squeeze()
            if err.ndim == 2:
                err = np.sum(err, axis=0)
            elif err.ndim > 2:
                raise ValueError("RAR: Error occured when calculate pde residue: err.ndim > 2")
            mean_err = np.mean(err)
            print(f'mean residual: {mean_err}')

            # [OPTIMIZABLE] 重采样选点策略
            new_points = select_resample_points(X, err, count)
            data.add_anchors(new_points)
            train(*args, **kwargs, disregard_previous_best=True, save_model=False)

    return wrapper
