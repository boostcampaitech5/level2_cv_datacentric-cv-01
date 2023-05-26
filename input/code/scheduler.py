import torch
from torch import cuda
from torch.optim import lr_scheduler
from model import EAST


class CustomScheduler:
    def __init__(self, scheduler_name, params: dict) -> None:
        try:
            scheduler_class = getattr(lr_scheduler, scheduler_name)
            self.scheduler = scheduler_class(**params)
        except AttributeError:
            print(
                "Error: '{}' scheduler does not exist in torch.optim.lr_scheduler.".format(
                    scheduler_name
                )
            )

    def step(self):
        self.scheduler.step()


def main():
    model = EAST()
    model.to("cuda" if cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150 // 2], gamma=0.1)
    params = {"optimizer": optimizer, "milestones": [150 // 2], "gamma": 0.1}

    scheduler = CustomScheduler(scheduler_name="MultiStepLR", params=params)
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()


if __name__ == "__main__":
    main()
