# Multi-Model QLoRA Fine-Tuning (RTX 4090)

This repository provides a reusable QLoRA fine-tuning pipeline supporting:
- google/gemma-3-4b-it
- mistralai/Ministral-3-3B-Instruct-2512

Dataset:
- teknium/OpenHermes-2.5

Design Goals:
- Single GPU (RTX 4090)
- QLoRA (4-bit) fine-tuning
- Scalable to 16B-class models
- No environment hardcoding

All training, model, and output parameters are configurable.
