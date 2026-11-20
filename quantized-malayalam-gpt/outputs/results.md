# Results

CPU only (AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD), torch 2.15.0+cpu, int8 engine onednn, 1 thread. Size = saved state dict. Generation = 200 new tokens from `കേരളത്തിൽ`, batch 1; forward = one 32x128 batch; both are medians of interleaved rounds (`outputs/speed.md`). Val loss = 40 fixed batches of 32x128 from P2's val split, the same windows for every row. Δ val loss is against the FP32 baseline; for QAT the fair reference is the control row.

| method | size (MB) | Linear weights (MB) | generation tok/s | forward tok/s | val loss | Δ val loss | sample |
|---|---|---|---|---|---|---|---|
| FP32 baseline (P2 checkpoint) | 4.42 | 3.23 | 149 | 26,285 | 1.5087 |  | കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്ക… |
| Dynamic int8 | 2.07 | 0.88 | 71 | 32,332 | 1.5099 | +0.0012 | കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നത്. ഒന്നാം സുമങ്ക… |
| Static int8, calibrated on real text | 2.23 | 0.98 | 50 | 13,035 | 1.5119 | +0.0032 | കേരളത്തിൽനിന്നും സംഭവം കഴിഞ്ഞ ആരോപണം യുവാഗ്രഹിക്കുന്നതും ഒന്നും സുമങ്ക… |
| Static int8, calibrated on torch.randn | 2.23 | 0.98 | 49 | 13,830 | 1.5112 | +0.0024 | കേരളത്തിൽനിന്നും സംഭവം കാട്ട് ആരോപണം യുവാഗ്രഹിക്കുന്നതു തിരുവനന്തപുരം … |
| QAT int8 (300-step fake-quant fine-tune) | 2.23 | 0.98 | 47 | 13,750 | 1.4827 | -0.0260 (+0.0047 vs control) | കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങ… |
| Control: FP32, same 300-step fine-tune | 4.42 | 3.23 | 151 | 26,739 | 1.4780 | -0.0307 | കേരളത്തിൽനിന്നും സംഭവം കുടുംബശ്രീയ ഏജൻ. ബഗാനിലെ പത്ര പല തിന്റെ വിജയങ്ങ… |
