This folder is populated automatically by `python build_models.py --data <your_data_folder>`.

After running build_models.py once, you should see files like:
  - ind_bundles.pkl
  - BPNN_F1.pth, CNN_F1.pth, BiLSTM_F1.pth (and F2-F4)
  - wfan_means_flat.pkl, wfan_means_2d.pkl, wfan_means_seq.pkl
  - flange_top_models.pkl, flange_weights.pkl
  - fold_results.pkl, dep_results.pkl
  - bpnn_input_dims.pkl, metadata.pkl

Section 2 of the app (Competition Prediction) loads from this folder.
You can also generate these via Section 1 (Train Your Own Model) -> Save tab.
