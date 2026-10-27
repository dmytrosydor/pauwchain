# Калібрування E1 на Kaggle T4

Мета — виміряти похибку чесного вузла між іншим залізом: процесор x86 сервера Kaggle (майнер) і GPU Tesla T4
(валідатор). Це доповнює виміри CPU ↔ MPS на Mac (docs/results.md, E1). У хмарі запускається лише
Python-калібрування, жодних вузлів чи мережі (правила Kaggle забороняють майнінг).

1. На Mac зібрати архів:
   ```sh
   scripts/kaggle_bundle.sh
   ```
2. kaggle.com → Datasets → New Dataset → завантажити `build/kaggle/pauwchain-py.zip` (назва, наприклад, `pauwchain-py`).
3. Code → New Notebook → Add Input → цей датасет. Settings → Accelerator: GPU T4, Internet: On.
4. У клітинці ноутбука:
   ```python
   !cp -r /kaggle/input/pauwchain-py/* /kaggle/working/ && cd /kaggle/working && ls
   !cd /kaggle/working && python -m sim.calibrate --out runs/e1_t4 --devices cpu,cuda --ks 10,20,50
   ```
   Прогін займає кілька хвилин; MNIST завантажиться в `data/`.
5. Завантажити `runs/e1_t4/transitions.csv`, `summary.json`, `manifest.json` (права панель → Output) і покласти в
   `experiments/runs/e1_t4/` на Mac. Графік: `.venv/bin/python -m sim.plots e1 experiments/runs/e1_t4`.

Якщо Kaggle розпакує архів у підтеку, шлях у кроці 4 буде `/kaggle/input/pauwchain-py/pauwchain-py/`.
