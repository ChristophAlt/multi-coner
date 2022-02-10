import torch.nn as nn
from pytorch_ie.models.modules.mlp import MLP


class SplitEmbedding(nn.Module):
    def __init__(
        self,
        input_embeddings: nn.Embedding,
        additional_embeddings: nn.Embedding,
        threshold_value: int,
    ) -> None:
        super(SplitEmbedding, self).__init__()
        self.input_embeddings = input_embeddings
        self.additional_embeddings = additional_embeddings
        self.threshold_value = threshold_value

        in_dim = self.additional_embeddings.weight.shape[1]
        out_dim = self.input_embeddings.weight.shape[1]

        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        # self.projection = MLP(
        #     input_dim=in_dim,
        #     output_dim=out_dim,
        #     hidden_dim=384,
        #     num_layers=2,
        # )

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        additional_emb_mask = tokens >= self.threshold_value

        additional_emb_tokens = tokens[additional_emb_mask] - self.threshold_value

        tokens[additional_emb_mask] = 0

        input_embedding = self.input_embeddings(tokens)

        # additional_embedding = self.additional_embeddings(additional_emb_tokens)

        # projected_additional_embedding = self.projection(additional_embedding).to(input_embedding.dtype)

        # input_embedding[additional_emb_mask] = projected_additional_embedding

        return input_embedding
