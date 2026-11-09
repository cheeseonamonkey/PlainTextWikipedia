# Vendored CI dependencies

These files support GitHub Actions on Ubuntu 24.04 x86-64 with Python 3.12.
The wheelhouse contains the resolved CPU dependencies of `REQUIREMENTS.TXT`
and `requirements-encoder.txt`, including fastText, PyTorch and OpenVINO.
Wheel files include their respective distribution license information.

`bekko/` contains the OpenVINO model, tokenizer and configuration from
[hotchpotch/bekko-embedding-v1-a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)
at revision `c721113d59a1d91b447450324f51c4b3332c924a`; its upstream model
card identifies the MIT license. `bin/actionlint` is version 1.7.12; its
license is alongside it. Files larger than GitHub's limit are stored in verified chunks. Run
`python3 scripts/restore_vendor.py` after checkout to reconstruct them.

Install offline with:

```sh
python -m pip install --no-index --find-links vendor/wheels -r REQUIREMENTS.TXT -r requirements-encoder.txt
```
 When updating requirements,
replace the wheelhouse with a complete compatible, resolved set and run the
real-model smoke and regression suite on the target platform. The source corpus,
publishing services, GitHub Actions and initial repository checkout still need
network access.
