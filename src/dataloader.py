import torch.utils.data as data
import torch

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, prot_dict, drug_seq_dict, drug_graph_dict):
        self.list_IDs = list_IDs
        self.df = df
        self.prot_dict = prot_dict
        self.drug_seq_dict = drug_seq_dict
        self.drug_graph_dict = drug_graph_dict

        sample_prot = (
            next(iter(prot_dict.values())) if prot_dict else torch.zeros(1280, 1200)
        )
        sample_drug_seq = (
            next(iter(drug_seq_dict.values()))
            if drug_seq_dict
            else torch.zeros(768, 100)
        )
        sample_drug_graph = (
            next(iter(drug_graph_dict.values()))
            if drug_graph_dict
            else torch.zeros(64, 100)
        )

        self.prot_dim = sample_prot.shape
        self.drug_seq_dim = sample_drug_seq.shape
        self.drug_graph_dim = sample_drug_graph.shape

        if (
            hasattr(self.df, "iloc")
            and self.list_IDs is not None
            and len(self.list_IDs) > 0
        ):
            sample_smiles_set = set(self.df.iloc[self.list_IDs]["SMILES"].values)
            if self.drug_graph_dict is not None and len(self.drug_graph_dict) > 0:
                missing_hgnn = [
                    s for s in sample_smiles_set if s not in self.drug_graph_dict
                ]
                if len(missing_hgnn) > 0:
                    missing_rate = (
                        len(missing_hgnn) / max(len(sample_smiles_set), 1) * 100.0
                    )
                    print(
                        f"⚠️ [数据加载警报] HGNN 图网络特征字典中丢失 {len(missing_hgnn)}/{len(sample_smiles_set)} (缺失率: {missing_rate:.2f}%) 的化合物表征！此部分在运行中将全部退化为默认全 0 矩阵。"
                    )
                    if missing_rate > 50.0:
                        print(
                            f"   🔴 [致命预警] 超半数 HGNN 图特征断裂！请紧急自检新输入的 .pt 字典的主键形态（可能是标准化 canonical SMILES、Drug_ID 导致与原文 SMILES 首字母与转义不齐）。未命中样本示例：{missing_hgnn[:2]}"
                        )

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        idx = self.list_IDs[index]
        drug_name = self.df.iloc[idx]["SMILES"]
        protein_name = self.df.iloc[idx]["Protein"]
        y = self.df.iloc[idx]["Y"]

        prot_feat = self.prot_dict.get(protein_name, torch.zeros(self.prot_dim))
        drug_seq_feat = self.drug_seq_dict.get(
            drug_name, torch.zeros(self.drug_seq_dim)
        )
        drug_graph_feat = self.drug_graph_dict.get(
            drug_name, torch.zeros(self.drug_graph_dim)
        )

        return prot_feat, drug_seq_feat, drug_graph_feat, y, drug_name, protein_name
