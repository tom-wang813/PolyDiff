from typing import Optional, Tuple

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions


class BertDiffusionModel(BertPreTrainedModel):
    """
    A BERT-based model for discrete diffusion, adapted to the Hugging Face
    `BertPreTrainedModel` framework.

    This model uses the standard BERT architecture and is designed to be
    initialized from a configuration object.
    The forward pass is modified to accept timestep embeddings, making it
    suitable for diffusion tasks.
    """

    def __init__(self, config: BertConfig):
        """
        Initializes the BertDiffusionModel.

        Args:
            config (BertConfig): The configuration for the BERT model.
                                 This object contains all the model architecture
                                 parameters (e.g., vocab_size, hidden_size,
                                 num_hidden_layers).
        """
        super().__init__(config)
        self.bert = BertModel(config)
        # Initialize weights for the new model as per the base class method
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        timestep_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions | Tuple[torch.Tensor, ...]:
        """
        Forward pass for the BertDiffusionModel.

        The forward pass is augmented to include `timestep_embeds`, which are added to the standard
        BERT sequence output. This allows the model to be conditioned on the diffusion timestep.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs. Shape:
                                                (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Mask to avoid performing
                                                    attention on padding token
                                                    indices.
            timestep_embeds (torch.Tensor, optional): Embeddings for the
                                                      diffusion timestep.
                                                      Shape: (batch_size,
                                                      hidden_size).
            ... (other standard BERT arguments)

        Returns:
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions:
            The standard BERT model output, where `last_hidden_state` has been
            conditioned on the timestep.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Standard BERT forward pass
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Add timestep embeddings to the sequence output
        if timestep_embeds is not None:
            # Expand timestep embeddings to match the sequence length dimension
            expanded_timestep_embeds = timestep_embeds.unsqueeze(1).expand(
                -1, sequence_output.size(1), -1
            )
            sequence_output = sequence_output + expanded_timestep_embeds

        if not return_dict:
            # If not returning a dict, update the first element of the output tuple
            outputs = (sequence_output,) + outputs[1:]
            return outputs

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=outputs.pooler_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def get_embedding_layer(self) -> nn.Module:
        """
        Returns the embedding layer of the underlying BERT model.

        This can be useful for accessing or modifying the embeddings directly,
        or for tasks like visualizing embedding spaces.

        Returns:
            torch.nn.Module: The embedding layer of the BERT model.
        """
        return self.bert.embeddings
