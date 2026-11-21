import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs import get_cfg_defaults
from dataloader import DTIDataset
from trace_dti_model import TraceDTIInferenceModel


def scenario_directory(scenario):
    return {
        "cv1": "S1_Random",
        "s2": "S2_Cold_Prot",
        "s3": "S3_Cold_Drug",
        "s4": "S4_Dual_Cold",
    }[scenario]


def find_checkpoint(checkpoint_dir, dataset, scenario, fold):
    result_split = scenario_directory(scenario)
    candidates = [
        checkpoint_dir / f"{dataset}_{scenario}_random_{fold}.pt",
        checkpoint_dir / f"random_{fold}.pt",
        checkpoint_dir / f"random_{fold}" / f"random_{fold}.pt",
        ROOT / "results"  / dataset / "TRACE-DTI" / result_split / f"random_{fold}" / f"random_{fold}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for fold {fold}. Searched: "
        + ", ".join(str(path) for path in candidates)
    )


def metrics(labels, scores):
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    f1_curve = (
        2 * precision_curve * recall_curve
        / (precision_curve + recall_curve + 1e-8)
    )
    best_index = int(np.argmax(f1_curve[:-1]))
    threshold = float(thresholds[best_index])
    predictions = (scores >= threshold).astype(int)
    has_both_classes = len(np.unique(labels)) > 1
    return {
        "auroc": roc_auc_score(labels, scores) if has_both_classes else np.nan,
        "auprc": average_precision_score(labels, scores) if has_both_classes else np.nan,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": float(f1_curve[best_index]),
        "mcc": matthews_corrcoef(labels, predictions),
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRACE-DTI checkpoints on prepared test PT files")
    parser.add_argument("--dataset", default="Kinase")
    parser.add_argument("--scenario", choices=["cv1", "s2", "s3", "s4"], default="cv1")
    parser.add_argument("--fold", type=int, default=0, help="1-5 for one fold; 0 evaluates all folds")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.fold not in range(0, 6):
        parser.error("--fold must be between 0 and 5")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    data_dir = args.output_dir or ROOT / "ours" / "TRACE-DTI" / "checkpoint" / scenario_directory(args.scenario)
    checkpoint_dir = args.checkpoint_dir or data_dir
    data_dir = data_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    cfg = get_cfg_defaults()
    results = []

    for fold in ([args.fold] if args.fold else range(1, 6)):
        pt_path = data_dir / f"{args.dataset}_{args.scenario}_test{fold}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Prepared test PT not found: {pt_path}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        test = payload["test"].reset_index(drop=True)
        dataset = DTIDataset(
            test.index.values,
            test,
            payload["prot_dict"],
            payload["drug_seq_dict"],
            payload["drug_graph_dict"],
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        first_prot = next(iter(payload["prot_dict"].values()))
        first_seq = next(iter(payload["drug_seq_dict"].values()))
        first_graph = next(iter(payload["drug_graph_dict"].values()))
        checkpoint_path = find_checkpoint(checkpoint_dir, args.dataset, args.scenario, fold)
        model = TraceDTIInferenceModel(
            prot_channels=first_prot.shape[0],
            drug_channels=first_seq.shape[0],
            hgnn_dim=first_graph.shape[1],
            embed_dim=cfg.SOLVER.EMBED_DIM,
            mlp_hidden_dim=cfg.DECODER.HIDDEN_DIM,
            dropout=cfg.SOLVER.DROPOUT,
        ).to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "Checkpoint is missing parameters required by the current model: "
                + ", ".join(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print(
                f"Ignoring {len(incompatible.unexpected_keys)} checkpoint keys "
                "not used by the current inference model."
            )
        model.eval()
        score_list = []
        with torch.inference_mode():
            for prot, seq, graph, _, _, _ in loader:
                logits, _, _ = model(prot.to(device), seq.to(device), graph.to(device))
                score_list.extend(torch.sigmoid(logits.squeeze(1).float()).cpu().tolist())

        labels = test["Y"].astype(int).to_numpy()
        scores = np.asarray(score_list, dtype=float)
        fold_metrics = metrics(labels, scores)
        results.append({"fold": fold, **fold_metrics})
        test_result = test.copy()
        test_result["score"] = scores
        test_result["prediction"] = (scores >= fold_metrics["threshold"]).astype(int)
        test_result["fold"] = fold
        test_result["threshold"] = fold_metrics["threshold"]
        test_result["checkpoint"] = str(checkpoint_path)
        output = data_dir / f"info_test{fold}.csv"
        test_result.to_csv(output, index=False)
        print(f"Fold {fold}: checkpoint={checkpoint_path}; predictions -> {output}")

    summary = pd.DataFrame(results)
    summary.to_csv(data_dir / "info_metrics.csv", index=False)
    print(f"Metrics -> {data_dir / 'info_metrics.csv'}")


if __name__ == "__main__":
    main()
