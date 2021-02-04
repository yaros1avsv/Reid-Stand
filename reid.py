import torch
import torchreid


def make_reid(visrank):
    datamanager = torchreid.data.ImageDataManager(
        root='static',
        sources='center',
        targets='center',
        height=256,
        width=128
    )

    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
    )
    torchreid.utils.load_pretrained_weights(model, 'static/reid/model.pth.tar-50')
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir='static/reid',
        max_epoch=50,
        eval_freq=10,
        print_freq=10,
        visrank_topk=visrank,
        test_only=True,
        visrank=True,
        rerank=False
    )


if __name__ == "__main__":
    make_reid(5)
