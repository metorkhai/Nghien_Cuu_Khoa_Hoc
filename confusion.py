"""
Compute multi-label confusion statistics for SoftLogic ViBERT models.

Usage:
	python -m softlogic_vibert.confusion confusion --model <path_to_model.pt>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ModelConfig
from .data import SentimentDataset, collate_batch, load_data_file
from .model import SoftLogicViBERT


def _load_checkpoint(
	model_path: str, device: torch.device
) -> Tuple[SoftLogicViBERT, object, List[str], ModelConfig]:
	checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

	if "model_config" in checkpoint:
		model_config = ModelConfig(**checkpoint["model_config"])
	else:
		model_config = ModelConfig()

	if "model" in checkpoint and isinstance(checkpoint["model"], torch.nn.Module):
		model = checkpoint["model"]
	else:
		model = SoftLogicViBERT(model_config)
		if "state_dict" in checkpoint:
			model.load_state_dict(checkpoint["state_dict"], strict=False)

	model = model.to(device)
	model.eval()

	tokenizer_name = checkpoint.get("tokenizer_name", model_config.model_name)
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

	label_list = checkpoint.get("label_list", [])

	return model, tokenizer, label_list, model_config


def _compute_confusion(
	logits: torch.Tensor, labels: torch.Tensor, threshold: float
) -> Dict[str, torch.Tensor]:
	probs = torch.sigmoid(logits)
	preds = (probs >= threshold).float()

	tp = (preds * labels).sum(dim=0)
	fp = (preds * (1 - labels)).sum(dim=0)
	fn = ((1 - preds) * labels).sum(dim=0)
	tn = ((1 - preds) * (1 - labels)).sum(dim=0)

	p = labels.sum(dim=0)
	n = (1 - labels).sum(dim=0)
	pp = preds.sum(dim=0)

	return {
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"tn": tn,
		"p": p,
		"n": n,
		"pp": pp,
	}


def _print_table(label_list: List[str], stats: Dict[str, torch.Tensor]) -> None:
	headers = ["label", "P", "N", "PP", "TP", "FP", "TN", "FN"]
	widths = [18, 8, 8, 8, 8, 8, 8, 8]

	def _fmt_row(values: List[str]) -> str:
		return " ".join(v.ljust(w) for v, w in zip(values, widths))

	print(_fmt_row(headers))
	print("-" * (sum(widths) + len(widths) - 1))

	for idx, label in enumerate(label_list):
		row = [
			label[:16],
			str(int(stats["p"][idx].item())),
			str(int(stats["n"][idx].item())),
			str(int(stats["pp"][idx].item())),
			str(int(stats["tp"][idx].item())),
			str(int(stats["fp"][idx].item())),
			str(int(stats["tn"][idx].item())),
			str(int(stats["fn"][idx].item())),
		]
		print(_fmt_row(row))


def run_confusion(args: argparse.Namespace) -> None:
	device = torch.device(
		args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
	)
	model, tokenizer, label_list, model_config = _load_checkpoint(args.model, device)

	if not label_list:
		raise RuntimeError("Label list not found in checkpoint. Cannot map confusion stats.")

	label_map = {lab: i for i, lab in enumerate(label_list)}

	data_path = args.data_path
	rows = load_data_file(data_path)

	dataset = SentimentDataset(
		rows,
		tokenizer,
		label_map,
		max_len=args.max_len or model_config.max_len,
	)
	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_batch,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	all_logits = []
	all_labels = []

	special_token_ids = torch.tensor(
		[tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
		device=device,
	)

	with torch.no_grad():
		for batch in loader:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			token_type_ids = batch.get("token_type_ids")
			if token_type_ids is not None:
				token_type_ids = token_type_ids.to(device)
			labels = batch["labels"].to(device)
			prag = batch.get("prag_features")
			if prag is not None:
				prag = prag.to(device)

			logits = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
				prag_features=prag,
				special_token_ids=special_token_ids,
				return_extras=False,
			)

			all_logits.append(logits.cpu())
			all_labels.append(labels.cpu())

	logits = torch.cat(all_logits, dim=0)
	labels = torch.cat(all_labels, dim=0)

	stats = _compute_confusion(logits, labels, args.threshold)
	_print_table(label_list, stats)

	if args.save_json:
		out = {}
		for idx, label in enumerate(label_list):
			out[label] = {
				"P": int(stats["p"][idx].item()),
				"N": int(stats["n"][idx].item()),
				"PP": int(stats["pp"][idx].item()),
				"TP": int(stats["tp"][idx].item()),
				"FP": int(stats["fp"][idx].item()),
				"TN": int(stats["tn"][idx].item()),
				"FN": int(stats["fn"][idx].item()),
			}
		with open(args.save_json, "w", encoding="utf-8") as f:
			json.dump(out, f, indent=2)
		print(f"Saved confusion stats to: {args.save_json}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compute multilabel confusion stats for SoftLogic ViBERT",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	confusion_parser = subparsers.add_parser("confusion", help="Compute confusion stats")
	confusion_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
	confusion_parser.add_argument(
		"--data-path", type=str, default="output_data.json", help="Path to data JSON/JSONL"
	)
	confusion_parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
	confusion_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
	confusion_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
	confusion_parser.add_argument("--device", type=str, default=None, help="Device override (cuda or cpu)")
	confusion_parser.add_argument("--max-len", type=int, default=None, help="Override max sequence length")
	confusion_parser.add_argument("--save-json", type=str, default=None, help="Optional JSON output path")

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.command == "confusion":
		run_confusion(args)


if __name__ == "__main__":
	main()
