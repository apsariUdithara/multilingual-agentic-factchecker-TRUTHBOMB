# 🚀 Traffic Law Pipeline - Complete System

## Overview

This is a **production-ready, all-in-one Jupyter notebook** for fine-tuning Phi-3.5-mini on traffic law fact-checking using **Unsloth + LoRA + RAG**.

**Perfect for**: Google Colab T4 GPU (free tier)  
**Training Time**: ~30-60 minutes  
**Model Size**: ~300MB (vs 7GB full model)  

---

## 📦 What's Included?

```
pipeline-setup/
├── traffic_law_pipeline.ipynb          ← MAIN NOTEBOOK (11 cells)
├── PIPELINE_GUIDE.md                   ← Comprehensive documentation
├── UPLOAD_DATASET_GUIDE.md             ← How to upload dataset to Drive
├── QUICK_REFERENCE.md                  ← One-page cheat sheet
├── README.md                           ← This file
├── data/
│   ├── traffic_law_dataset.jsonl       ← Training data
│   └── (Motor-Traffic-Consolidated-2024.pdf - optional)
├── models/
│   └── phi3-traffic-lora/              ← Trained LoRA adapters
├── vector_db/                          ← ChromaDB embeddings
└── src/ (deprecated - now in notebook)
    ├── generate_dataset.py
    ├── fine_tune.py
    ├── build_vector_db.py
    └── rag_inference.py
```

---

## 🎯 Quick Start (5 Minutes)

### On Google Colab:

1. **Open notebook**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File > Open from URL
   - Paste: `https://colab.research.google.com/drive/...` (your Colab link)

2. **Select GPU**
   - Runtime > Change runtime type > T4 GPU
   - Click Save

3. **Mount Google Drive**
   - Run Cell 1 (takes 30 seconds)

4. **Run Full Pipeline**
   - Execute cells in order (Cell 1 → Cell 8)
   - Takes ~30-45 minutes total

5. **Test Results**
   - Run Cell 7 (Test Custom Claims)
   - Modify claims as needed

✅ **Done!** Your model is trained and saved to Google Drive

---

## 🔄 What Each Cell Does

| Cell | Name | Purpose | Time |
|------|------|---------|------|
| 1 | Environment | Mount Drive, install packages | 5 min |
| 2 | Dataset | Load/generate JSONL from PDF | 2 min |
| 3 | Fine-Tuning | Train with Unsloth + LoRA | 20 min |
| 4 | Vector DB | Build ChromaDB from PDF | 2 min |
| 5 | Inference | Load model for testing | 1 min |
| 6 | Test Pipeline | Verify with sample claims | 1 min |
| 7 | Custom Test | Test your own claims | variable |
| 8 | Fast Load | Load existing model (skip train) | 2 min |

---

## 📋 Prerequisites

✅ Google account  
✅ Google Drive (15GB free)  
✅ Colab (free tier works!)  
✅ Local dataset (`traffic_law_dataset.jsonl`)  
✅ Optional: Legal PDF for vector DB  

---

## 🚀 Two Workflows

### Workflow A: Complete Training (First Time)

```
Start → Env Setup → Load Dataset → Train Model → Build DB → Test
   ↓       ↓          ↓           ↓            ↓        ↓
   5min    2min       20min       2min        1min     5min
                                  └─ Save to Drive ─┘
```

**Total Time**: ~35 minutes

### Workflow B: Fast Inference (Subsequent Times)

```
Start → Mount Drive → Load Model → Test Claims
   ↓       ↓           ↓           ↓
  30sec   30sec       1min        1min
   (skip training entirely!)
```

**Total Time**: ~3 minutes

---

## 💾 File Management

### Upload Dataset to Google Drive

**Before running notebook:**

1. Open [Google Drive](https://drive.google.com)
2. Create folder: `traffic_law_pipeline`
3. Create subfolder: `data`
4. Upload your `traffic_law_dataset.jsonl`

See **UPLOAD_DATASET_GUIDE.md** for detailed instructions.

### What Gets Created

After running notebook, Google Drive will contain:

```
traffic_law_pipeline/
├── data/
│   ├── traffic_law_dataset.jsonl     (your upload)
│   └── Motor-Traffic-Consolidated-2024.pdf  (optional)
├── models/
│   └── phi3-traffic-lora/
│       ├── adapter_model.bin         (300MB - your trained adapters!)
│       ├── adapter_config.json
│       ├── tokenizer.model
│       └── [other files]
├── vector_db/
│   ├── chroma.sqlite3
│   └── [embeddings]
└── outputs/
    ├── training_checkpoints/
    └── test_results.json
```

---

## 🔧 Configuration

### Dataset Path
```python
GDRIVE_PROJECT_PATH = "/content/gdrive/My Drive/traffic_law_pipeline"
DATASET_PATH = f"{GDRIVE_PROJECT_PATH}/data/traffic_law_dataset.jsonl"
```

### Training Hyperparameters
```python
max_seq_length = 2048              # Context length
num_train_epochs = 3               # Training epochs
learning_rate = 2e-4               # Learning rate
per_device_train_batch_size = 2    # Batch size
r = 16                             # LoRA rank
```

### Inference Settings
```python
max_new_tokens = 512               # Max output length
k = 3                              # Retrieved documents
```

---

## 📊 Performance

| Aspect | Value |
|--------|-------|
| **Base Model** | Phi-3.5-mini (3.8B params) |
| **Quantization** | 4-bit (INT4) |
| **Fine-tuning Method** | LoRA |
| **LoRA Rank** | 16 |
| **Training Method** | SFT (Supervised Fine-Tuning) |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB |
| **Training Speed** | 2-5x faster than standard |
| **Memory** | 70% reduction |
| **Model Size** | ~300MB (adapters only) |
| **Inference Speed** | 2x faster |

---

## 🎓 Learning the System

### Level 1: Just Run It
- Open notebook in Colab
- Click "Run all cells"
- Watch it work

### Level 2: Understand Each Part
- Read comments in notebook cells
- Check **PIPELINE_GUIDE.md**
- Modify test claims (Cell 7)

### Level 3: Customize It
- Upload your own dataset
- Adjust hyperparameters
- Integrate with your system

### Level 4: Production Deploy
- Export adapters from Drive
- Build inference API
- Scale with vLLM/TorchServe

---

## 🐛 Troubleshooting

### "Dataset not found"
→ See **UPLOAD_DATASET_GUIDE.md**

### "Out of Memory"
→ Reduce batch size or max_seq_length in Cell 3

### "Module not found"
→ Run Cell 1 again to reinstall packages

### "Model not found in Google Drive"
→ Ensure training completed; check outputs folder

### PDF won't process
→ Ensure format is PDF; notebook creates sample if missing

See **PIPELINE_GUIDE.md** for more troubleshooting.

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **PIPELINE_GUIDE.md** | Comprehensive manual with all details |
| **UPLOAD_DATASET_GUIDE.md** | Step-by-step dataset upload instructions |
| **QUICK_REFERENCE.md** | One-page cheat sheet for quick lookup |
| **README.md** | This overview document |

---

## 🔗 Important Links

- [Google Colab](https://colab.research.google.com/)
- [Google Drive](https://drive.google.com/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [Hugging Face Hub](https://huggingface.co/)

---

## 📈 Optimization Tips

### For Speed:
- Use packing: `packing=True`
- Reduce epochs: `num_train_epochs = 1`
- Increase batch size (if VRAM allows)

### For Accuracy:
- More epochs: `num_train_epochs = 5+`
- Lower learning rate: `learning_rate = 1e-4`
- Larger LoRA rank: `r = 32+`
- More training data

### For Memory:
- 4-bit quantization ✅ (already enabled)
- Gradient checkpointing ✅ (already enabled)
- Reduce batch size
- Reduce max_seq_length

---

## 🎯 Use Cases

### 1. Legal Document Fact-Checking
- Train on traffic laws
- Test claims against legal context
- Provide verdict with confidence

### 2. Multi-Domain Extension
- Replace traffic PDF with other legal docs
- Retrain adapters (30 min)
- Deploy new expert system

### 3. Integration Point
- Load model in Python API
- Accept claims via REST endpoint
- Return JSON verdicts
- Scale with vLLM

### 4. Research/Testing
- Experiment with hyperparameters
- Compare different training data
- Benchmark RAG quality
- Publish results

---

## ✅ Implementation Checklist

### Before First Run:
- [ ] Google account ready
- [ ] Downloaded local dataset
- [ ] Created Google Drive folder structure
- [ ] Uploaded `traffic_law_dataset.jsonl`
- [ ] Opened notebook in Colab

### During Training:
- [ ] Monitor training loss (should decrease)
- [ ] Check GPU utilization
- [ ] Watch for error messages
- [ ] Note training duration

### After Training:
- [ ] Verify model saved to Google Drive
- [ ] Test with sample claims
- [ ] Download adapters for backup
- [ ] Document any customizations

### For Production:
- [ ] Validate on 50+ diverse claims
- [ ] Test on new, unseen data
- [ ] Build inference wrapper
- [ ] Set up error logging
- [ ] Monitor latency/throughput

---

## 🔄 Iteration Workflow

### Iteration 1: Get Working
```
Upload dataset → Run full notebook → Test → Success ✓
```

### Iteration 2: Improve
```
Adjust hyperparameters → Retrain → Compare results → Success ✓
```

### Iteration 3: Production
```
Export model → Build API → Deploy → Monitor → Scale ✓
```

---

## 📞 Support

1. **Check Documentation**
   - PIPELINE_GUIDE.md (comprehensive)
   - QUICK_REFERENCE.md (quick lookup)
   - UPLOAD_DATASET_GUIDE.md (data setup)

2. **Review Notebook Comments**
   - Each cell has detailed explanations
   - Error messages are descriptive
   - Progress updates show what's happening

3. **Debug Issues**
   - Run cells individually
   - Check file paths match Google Drive
   - Verify dataset format

---

## 🎉 Success Indicators

✅ **Step 1**: Cell 1 installs packages without errors  
✅ **Step 2**: Dataset loads (Cell 2 shows count)  
✅ **Step 3**: Training starts (loss values appear)  
✅ **Step 4**: Model saves to Google Drive  
✅ **Step 5**: Test claim returns verdict  
✅ **Step 6**: Results can be downloaded  

**If all ✅**, congratulations! System is working perfectly.

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Apr 2026 | Initial: 5 separate Python files |
| 2.0 | May 5, 2026 | **Current**: Single optimized notebook + Google Drive |

---

## 📄 License

This pipeline is provided as-is for educational and research purposes.

---

## 🙏 Credits

Built with:
- **Unsloth** - Faster fine-tuning
- **Phi-3.5-mini** - Efficient language model
- **LoRA** - Parameter-efficient adaptation
- **ChromaDB** - Vector embeddings
- **LangChain** - RAG orchestration
- **Hugging Face** - Transformers library

---

## 🚀 Next Steps

1. **Start**: Open `traffic_law_pipeline.ipynb` in Google Colab
2. **Setup**: Follow Cell 1 instructions
3. **Train**: Execute cells 2-5 in order
4. **Test**: Run Cell 6 to verify
5. **Use**: Run Cell 7 with your custom claims

**You're ready!** 🎯

---

**Questions?** Check the detailed guides or review the notebook comments.

**Last Updated**: May 5, 2026  
**System**: Optimized for Google Colab T4 GPU  
**Framework**: PyTorch + Transformers + Unsloth  
