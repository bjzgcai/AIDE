import argparse
import json
import os
import re

from rdkit import Chem
from tqdm import tqdm


def extract_smiles(text):
    # 更完整的 SMILES 可能包含的字符集合
    smiles_charset = set("BCNOFPSIKcbnosiph*[]()=+-#@$\\/123456789%.,:<>|~")
    multi_char_elements = {"Br", "Cl"}  # 需要特殊处理的双字符元素

    # 正则表达式提取潜在的 SMILES 片段
    candidate_smiles = re.findall(r"\b[a-zA-Z0-9@#=+\-\\\/\[\]\(\)%]+\b", text)

    valid_candidates = []
    for s in candidate_smiles:
        # 检查字符串是否只包含允许的字符或已知的双字符元素
        i, valid = 0, True
        while i < len(s):
            if s[i : i + 2] in multi_char_elements:  # 处理 "Br"、"Cl"
                i += 2
            elif s[i] in smiles_charset:
                i += 1
            else:
                valid = False
                break
        if valid:
            valid_candidates.append(s)

    # 使用 RDKit 进行合法性验证
    valid_smiles = [s for s in valid_candidates if Chem.MolFromSmiles(s) is not None]

    return valid_smiles


def is_valid_smiles(smiles):
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return True


def extract_dna_sequences(text, min_length=10):
    dna_pattern = rf"[ATCGN]{{{min_length},}}"
    matches = re.findall(dna_pattern, text, re.IGNORECASE)
    return matches


def extract_rna_sequences(text, min_length=10):
    rna_pattern = rf"[AUCGN]{{{min_length},}}"
    matches = re.findall(rna_pattern, text)
    return matches


def extract_protein_sequences(text, min_length=20):
    sequence_pattern = rf"[ARNDCEQGHILKMFPSTWYV]{{{min_length},}}"
    matches = re.findall(sequence_pattern, text)
    return matches


def post_process(input_dir: str, output_dir: str):
    """
    Post process the generated data.
    """
    datasets = os.listdir(f"{input_dir}")
    for dataset in tqdm(datasets):
        tasks = os.listdir(f"{input_dir}/{dataset}")
        for task in tasks:
            files = os.listdir(f"{input_dir}/{dataset}/{task}")
            for file in files:
                if file == "raw":
                    continue
                with open(f"{input_dir}/{dataset}/{task}/{file}", "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))
                converted_data = []
                for item in tqdm(data):
                    if len(item.split("\t")) != 2:
                        item = item.replace("\\t", "\t")
                        if len(item.split("\t")) != 2:
                            # print(f"{input_dir}/{dataset}/{task}/{file}")
                            # print(item.split("\t"))
                            # return
                            continue
                    smiles = extract_smiles(item)
                    # print(smiles)
                    dna = extract_dna_sequences(item)
                    # print(dna)
                    rna = extract_rna_sequences(item)
                    # print(rna)
                    protein = extract_protein_sequences(item)
                    # print(protein)
                    # return
                    for smi in smiles:
                        item = item.replace(smi, "<mol>" + smi + "</mol>")
                    for d in dna:
                        item = item.replace(d, "<dna>" + d + "</dna>")
                    for r in rna:
                        item = item.replace(r, "<rna>" + r + "</rna>")
                    for p in protein:
                        item = item.replace(p, "<protein>" + p + "</protein>")
                    if len(item.split("\t")[-1]) != 0:
                        converted_data.append(item)
                os.makedirs(f"{output_dir}", exist_ok=True)
                filename = file.replace(".json", ".tsv")
                prompt = input_dir.split("/")[-1]
                with open(
                    f"{output_dir}/{prompt}_{dataset}_{task}_{filename}", "w"
                ) as f:
                    f.write("\n".join(converted_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.expanduser("~/AIDE/processed_data/virus"),
        help="input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/AIDE/post_processed_data/virus_new_v2"),
        help="output directory",
    )
    args = parser.parse_args()
    post_process(args.input_dir, args.output_dir)
