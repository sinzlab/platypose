import os
from unittest.mock import MagicMock

import wandb

env = os.environ.get("ENV", "dev")
platform = {"dev": MagicMock(), "wandb": wandb}[env]
