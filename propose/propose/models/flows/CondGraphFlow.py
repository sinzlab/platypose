import propose.poses
import torch
import wandb
from propose.models.distributions import StandardNormal
from propose.models.flows.GraphFlow import GraphFlow
from propose.models.nn.CondGNN import CondGNN
from propose.models.nn.embedding import embeddings
from propose.models.transforms.transform import (GraphActNorm,
                                                 GraphAffineCouplingTransform,
                                                 GraphCompositeTransform)
from propose.poses.human36m import Human36mPose, MPIIPose
from torch_geometric.data import HeteroData


class CondGraphFlow(GraphFlow):
    def __init__(
        self,
        features=3,
        num_layers=5,
        context_features=2,
        hidden_features=100,
        embedding_net=None,
        relations=None,
        use_attention=False,
        root_features=3,
        # mask_idx=[0, 2, 5, 8, 10, 12, 15]
    ):
        """
        Conditional Graph Flow model. The model is composed of a CondGNN and a GraphFlow.
        :param features: Number of features in the input.
        :param num_layers: Number of flow layers.
        :param context_features: Number of features in the context after embedding.
        :param hidden_features: Number of features in the hidden layers.
        :param embedding_net: (optional) Network to embed the context. default: nn.Identity
        :param relations: (optional) List of relations to use. default: None
        :param use_attention: (optional) Whether to use attention. default: False
        :param root_features: (optional) Number of features in the root node. default: 3
        """

        def create_net(in_features, out_features):
            return CondGNN(
                in_features=in_features,
                context_features=context_features,
                out_features=out_features,
                hidden_features=hidden_features,
                relations=relations,
                use_attention=use_attention,
                root_features=root_features,
            )

        coupling_constructor = GraphAffineCouplingTransform

        layers = []
        for i in range(num_layers):
            mask = -torch.ones(features)
            mask[
                i % features
            ] = 1  # iterate over feature pairs in a leave-one-out fashion

            layers.append(GraphActNorm(features=features))
            layers.append(
                coupling_constructor(mask=mask, transform_net_create_fn=create_net)
            )

        super().__init__(
            transform=GraphCompositeTransform(layers),
            distribution=StandardNormal([features]),
            embedding_net=embedding_net,
        )

    def forward(self, inputs: HeteroData) -> torch.Tensor:
        """
        The function takes in a tensor of inputs and returns the log probability of the inputs

        :param inputs: the input data, a tensor of size [batch_size, input_size]
        :return: The log probability of the inputs.
        """
        return self.log_prob(inputs)

    @classmethod
    def build_model(cls, config: dict) -> GraphFlow:
        """
        Builds a CondGraphFlow model from config
        :param config: Config dictionary
        :return: CondGraphFlow model
        """
        embedding_net = None
        if config["embedding"]:
            embedding_net = embeddings[config["embedding"]["name"]](
                **config["embedding"]["config"]
            )

        return cls(**config["model"], embedding_net=embedding_net)

    @classmethod
    def from_pretrained(cls, artifact_name: str) -> GraphFlow:
        """
        Constructs a pretrained model from the wandb model registry.
        :param artifact_name: Name of the artifact to load.
        :return: A pretrained model.
        """
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type="model")

        if wandb.run:
            wandb.run.use_artifact(artifact, type="model")

        flow = cls.build_model(artifact.metadata)

        artifact_dir = artifact.download()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        flow.load_state_dict(
            torch.load(artifact_dir + "/model.pt", map_location=torch.device(device)),
            strict=False,
        )

        return flow

    def set_device(self) -> bool:
        """
        If a GPU is available, move the model to the GPU
        :return: a boolean value.
        """
        if torch.cuda.is_available():
            self.to("cuda:0")
            return True

        return False

    @classmethod
    def preprocess(cls, pose_2d: propose.poses.Human36mPose) -> HeteroData:
        pose_2d.pose_matrix = pose_2d.pose_matrix * 0.0139
        pose_2d.pose_matrix = pose_2d.pose_matrix - pose_2d.pose_matrix[:, 0]
        pose_2d.pose_matrix[..., 1] = -pose_2d.pose_matrix[..., 1]

        return Human36mPose().conditional_graph(pose_2d)
