import torch
from ModuleGraph import get_func_graph
from torch_geometric.loader import DataLoader
from utils.common import get_model
import base64
import json
from pathlib import Path

def get_func_embeds(filename, fn_names):
    with torch.no_grad():
        graphs = get_func_graph(filename, fn_names, gen_vis=True, debug=False)

        loader = DataLoader(graphs, batch_size=2)
        model = get_model()

        embeds = []
        for batch in loader:
            embed_batch = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            for embed in embed_batch:
                embeds.append(embed.numpy())

        return embeds

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate A Geometric HeteroGraph from an IR File")
    parser.add_argument(
        "--input", "-i", dest="inp", required=True, help="A LLVM IR file we will lower to a HeteroGraph"
    )
    parser.add_argument(
        "--funcs", "-f", dest="funcs", required=True, help="The function which needs to be extracted"
    )
    args = parser.parse_args()

    filename = args.inp
    fn_names = args.funcs.split(",")

    embeds = get_func_embeds(filename, fn_names)
    print(embeds)

    embed_dict = {fn_name: embed.astype("float64").tolist() for fn_name, embed in zip(fn_names, embeds)}

    print(embed_dict)
    json_file = Path(filename).with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(embed_dict, f)

