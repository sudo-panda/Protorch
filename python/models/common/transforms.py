import warnings
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer

# Ideally should come from the digit_embeddings module, but to avoid llvm dependency issues, we define it here.
# from mltraining.digit_embeddings import get_embedding_from_lookup_output, feat_count

feat_count = 12000000

def get_embedding_from_lookup_output(token_ids_tensor, embeds):
    embedded = embeds(token_ids_tensor)  # shape: [batch_size, no. numbers, 2, max_len, 10]

    scaling = torch.tensor([1, 10], dtype=embedded.dtype, device=embedded.device).view(1, 1, 2, 1, 1)
    embedded_mult = embedded * scaling  # [batch_size, no. numbers, 2, max_len, 10]

    embedded_prefinal = embedded_mult[:, :, 0] * embedded_mult[:, :, 1]  # shape: [batch_size, no. numbers, max_len, 10]

    final_embedding_sum = torch.sum(embedded_prefinal, dim=-2)  # shape: [batch_size, no. numbers, 10]

    reduced_final_embedding = final_embedding_sum / (torch.max(torch.abs(final_embedding_sum), dim=-1, keepdim=True).values + 1)

    return reduced_final_embedding

class IdentityTransform(nn.Module):
    def __init__(self, input_shape):
        super(IdentityTransform, self).__init__()
        self.input_shape = input_shape

    def forward(self, node, label, token_ids):
        return node

    def get_embedding_dim(self):
        return self.input_shape[-1]

    def get_output_shape(self, node_shape):
        assert node_shape[-1] == self.input_shape[-1], \
            f"Expected node shape last dimension {node_shape[-1]} to match input dimension {self.input_shape[-1]}"
        
        # Return the same shape as input
        return node_shape

class DigitEmbedTransform(nn.Module):
    def __init__(self, input_shape, digit_embed_size):
        super(DigitEmbedTransform, self).__init__()
        self.digit_embedding = nn.Embedding(feat_count, digit_embed_size, padding_idx=0)
        self.input_shape = input_shape

    def forward(self, node, label, token_ids):
        """
         node: (batch * seq, feat)
         label: list of list of str (batch, seq)
        """
        emb_list = []
        for token_ids_tensor in token_ids:
            # token_ids_tensor is a set with a single tensor
            assert isinstance(token_ids_tensor, set) and len(token_ids_tensor) == 1
            lookup_output = list(token_ids_tensor)[0].to(self.digit_embedding.weight.device)
            emb = get_embedding_from_lookup_output(lookup_output, self.digit_embedding)
            emb = emb.view(*emb.shape[:-2], emb.shape[-1] * emb.shape[-2]) # (batch, feat * digit_embed_size)
            emb_list.append(emb)
        
        return torch.cat(emb_list, dim=0).to(device=node.device)

    def get_embedding_dim(self):
        return self.digit_embedding.embedding_dim * self.input_shape[-1]

    def get_output_shape(self, node_shape):
        assert node_shape[-1] == self.input_shape[-1], \
            f"Expected node shape last dimension {node_shape[-1]} to match input dimension {self.input_shape[-1]}"
        
        # Return the new shape with the embedding dimension
        return node_shape[:-1] + (self.get_embedding_dim(),)

class TextEmbedTransform(nn.Module):
    def __init__(self, input_shape, model="jinaai/jina-embeddings-v2-base-code"):
        super(TextEmbedTransform, self).__init__()
        self.input_shape = input_shape

        warnings.filterwarnings("ignore",
            message="optimum is not installed. To use OnnxConfig and BertOnnxConfig, "
                    "make sure that `optimum` package is installed")
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def forward(self, node, labels, token_ids):
        assert isinstance(labels, list) and all(isinstance(item, list) for item in labels)

        all_labels = sum(labels, [])
        
        return self.model.encode(
                all_labels,
                convert_to_tensor=True,
                device="cuda",
                show_progress_bar=False
            ).clone().to(node.device)

    def get_embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def get_output_shape(self, node_shape):
        assert node_shape[-1] == self.input_shape[-1], \
            f"TextEmbedTransform: Expected node shape last dim {node_shape[-1]} to match input last dim {self.input_shape[-1]}"

        return node_shape[:-1] + (self.get_embedding_dim(),)

class Transforms(nn.Module):
    def __init__(self, transforms, x_dict_shapes, digit_embed_size=64):
        super().__init__()
        self.transform_map = {
            "None": lambda shape: IdentityTransform(shape),
            "DigitEmbed": lambda shape: DigitEmbedTransform(shape, digit_embed_size),
            "TextEmbed": lambda shape: TextEmbedTransform(shape),
        }

        self.transforms = nn.ModuleDict(
            {
                name: self.transform_map[transform](x_dict_shapes[name])
                for name, transform in transforms.items()
            }
        )

    def forward(self, x_dict, labels, token_ids):
        for node_type, node_data in x_dict.items():
            x_dict[node_type] = self.transforms[node_type](node_data, labels[node_type], token_ids[node_type])

        return x_dict
    
    def get_output_dim(self):
        return {
            name: transform.get_embedding_dim() # type: ignore
            for name, transform in self.transforms.items()
        }

    def get_output_shape(self, x_shapes):
        return { 
            name: transform.get_output_shape(x_shapes[name]) # type: ignore
            for name, transform in self.transforms.items() 
        }

if __name__ == "__main__":
    attr_labels = [['convergent', 'mustprogress', 'noinline', 'norecurse', 'nounwind', 'optnone', 
                   '"frame-pointer"="all"', '"no-trapping-math"="true"', '"stack-protector-buffer-size"="8"', 
                   '"target-cpu"="sm_60"', '"target-features"="+ptx82,+sm_60"', '"uniform-work-group-size"="true"', 
                   'noundef', 'alwaysinline', 'nonnull', 'align 8', 'dereferenceable(12)', 'align 4', 
                   'dereferenceable(44)', 'nocallback', 'nofree', 'nosync', 'speculatable', 'willreturn', 
                   'memory(none)', 'dereferenceable(20)', 'align 1', 'dereferenceable(1)', 
                   'byval(%"struct.cuda::std::__4::plus")', 'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type")', 
                   'dereferenceable(4)', 'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.1")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.2")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.3")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.4")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.5")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.6")',
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.7")', 
                   'byval(%"struct.cub::CUB_200200___CUDA_ARCH_LIST___NS::Int2Type.8")'], 
                  ['convergent', 'mustprogress', 'noinline', 'nounwind', 'optnone', 
                   '"frame-pointer"="all"', '"no-trapping-math"="true"', '"stack-protector-buffer-size"="8"', 
                   '"target-cpu"="sm_70"', '"target-features"="+ptx82,+sm_70"', 'noundef', 'norecurse', 
                   '"uniform-work-group-size"="true"', 'alwaysinline', 'nocallback', 'nofree', 'nosync', 
                   'speculatable', 'willreturn', 'memory(none)'], 
                  ['convergent', 'mustprogress', 'noinline', 'nounwind', 'optnone', '"frame-pointer"="all"', 
                    '"no-trapping-math"="true"', '"stack-protector-buffer-size"="8"', '"target-cpu"="sm_70"', 
                    '"target-features"="+ptx82,+sm_70"', 'noundef', 'norecurse', '"uniform-work-group-size"="true"', 
                    'alwaysinline', 'nocallback', 'nofree', 'nosync', 'speculatable', 'willreturn', 'memory(none)'], 
                  ['convergent', 'mustprogress', 'noinline', 'nounwind', 'optnone', '"frame-pointer"="all"', 
                   '"no-trapping-math"="true"', '"stack-protector-buffer-size"="8"', '"target-cpu"="sm_70"', 
                   '"target-features"="+ptx82,+sm_70"', 'noundef', 'norecurse', '"uniform-work-group-size"="true"', 
                   'alwaysinline', 'nocallback', 'nofree', 'nosync', 'speculatable', 'willreturn', 'memory(none)']]
    
    attr_node_list = []
    for attrs in attr_labels:
        attr_node = torch.rand(len(attrs)).to(device="cuda")
        print("Node X:", attr_node.shape)
        attr_node_list.append(attr_node)
    attr_node = torch.cat(attr_node_list, dim=0).to(device="cuda")

    print("Node X Dict:", attr_node.shape)
    text_embed_transform = TextEmbedTransform(attr_node.shape)
    
    embeds_all = text_embed_transform(attr_node, attr_labels)
    print("Embeds:", embeds_all.shape)

    number_nodes = torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]).to(device="cuda")
    number_labels = [['1', '0', '2', '3', '4', '1.66666666666667', '399', '16'], ['1', '0', '399', '-1'], 
                     ['1', '0', '2', '399', '0.5'], ['1', '0', '399']]
    
    size_nodes = torch.tensor([[64.,  8., 64.], [32.,  4., 32.], [ 1.,  1.,  8.], [64.,  8., 64.], 
                             [32.,  4., 32.], [ 1.,  1.,  8.], [64.,  8., 64.], [32.,  4., 32.], 
                             [ 1.,  1.,  8.], [64.,  8., 64.], [32.,  4., 32.], [ 1.,  1.,  8.]]).to(device="cuda")
    size_labels = [['', '', ''], ['', '', ''], ['', '', ''], ['', '', '']]
    digit_embed_transform = DigitEmbedTransform(size_nodes.shape, 10)
    number_embeds = digit_embed_transform(number_nodes, number_labels)
    size_embeds = digit_embed_transform(size_nodes, size_labels)

    print("Number Embeds:", number_embeds.shape)
    print("Size Embeds:", size_embeds.shape)

    assert number_embeds.shape == number_nodes.shape[:1] + (digit_embed_transform.get_embedding_dim(),)
    assert size_embeds.shape == size_nodes.shape + (digit_embed_transform.get_embedding_dim(),)