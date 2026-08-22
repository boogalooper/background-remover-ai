# Third-party models and libraries

Background Remover AI does not bundle model weights. Weights are downloaded from their original Hugging Face repositories on first use.

- BRIA RMBG-2.0 — https://huggingface.co/briaai/RMBG-2.0 — non-commercial terms / CC BY-NC 4.0 as described by BRIA.
- BiRefNet HR Matting — https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting — MIT.
- BiRefNet Portrait — https://huggingface.co/ZhengPeng7/BiRefNet-portrait — see the model repository for current terms.

Core runtime libraries are installed from their official Python package indexes by `install.bat` and retain their respective licenses.

- tifffile — TIFF container writer used for safe RGBA TIFF export; installed from PyPI and retains its upstream license.
