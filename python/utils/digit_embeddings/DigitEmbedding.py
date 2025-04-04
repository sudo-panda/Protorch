import re
import torch
import torch.nn as nn
import pathlib

script_path = pathlib.Path(__file__).parent.resolve()

embed_size = 3

feat_count = 0
feature_file = open(str(script_path / "FeatureMap" / "feature_map_file_pg_plus_text_all_digits.txt"), 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_key = feature_key
        feature_map[feature_key] = feat_count
        feat_count = feat_count + 1
embeds = nn.Embedding(feat_count, embed_size)

def is_numeric(txt):
    if re.match(r"^[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$", txt):
        return True
    else:
        return False

def get_digit_and_pos(txt):
    digits_array = []
    digits_pos_array = []
    if is_numeric(txt) == True:
        point_position = txt.find('.')
        if point_position > -1:
            for i in range(0, point_position):
                digits_array.append(txt[i])
            for i in reversed(range(point_position)):
                digits_pos_array.append(str(i))
            j = -1
            for i in range(point_position+1, len(txt)):
                digits_array.append(txt[i])
                digits_pos_array.append(str(j))
                j = j - 1
        else:
            for i in range(0, len(txt)):
                digits_array.append(txt[i])
            for i in reversed(range(len(txt))):
                digits_pos_array.append(str(i))
    # print(digits_array)
    # print(digits_pos_array)
    return digits_array, digits_pos_array

def get_digit_emb_of_number(token):
    digits = []
    digits_pos = []
    digit_embedding_vector = []
    digit_pos_vector = []
    reduced_final_embedding = []
    if is_numeric(token) == True:
        digits, digits_pos = get_digit_and_pos(token)
        for digit in digits:
            if digit in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                digit_embedding_vector.append(node_embed)
            else:
                digit_embedding_vector.append(torch.Tensor([[0.0,] * embed_size]))

        digit_embedding_vector = torch.cat(digit_embedding_vector)

        for digit_pos in digits_pos:
            if digit_pos in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit_pos]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_mult = node_embed * 10
                digit_pos_vector.append(node_embed_mult)
            else:
                digit_pos_vector.append(torch.Tensor([[0.0,] * embed_size]))

        digit_pos_vector = torch.cat(digit_pos_vector)

        final_embedding = digit_embedding_vector * digit_pos_vector
        final_embedding_sum = torch.sum(final_embedding, axis=0)
        reduced_final_embedding = final_embedding_sum / (torch.max(torch.abs(final_embedding_sum)) + 1)
    else:
        reduced_final_embedding = torch.Tensor([0.0,] * embed_size)

    return reduced_final_embedding


def main():
    print(torch.cdist(get_digit_emb_of_number("1").unsqueeze(0), get_digit_emb_of_number("3").unsqueeze(0)))
    print(torch.cdist(get_digit_emb_of_number("2").unsqueeze(0), get_digit_emb_of_number("4").unsqueeze(0)))

if __name__ == "__main__":
    main()