# Demo

Watch the trained network classify a light curve: after every new observation it outputs an updated
probability that the object is a Tidal Disruption Event. It loads the final trained model, trains nothing,
and needs no dataset — five example light curves are included.

```bash
# Clone locally the folders /tde , /outputs/final_test and /demo (this one), maintaining folder hierarchy
git clone --filter=blob:none --sparse https://github.com/sparrow-k/TDE-Classification.git && cd TDE-Classification && git sparse-checkout set demo tde outputs/final_test && cd demo
# Make sure you have the following dependencies:
pip install torch numpy pandas matplotlib scikit-learn
# from within the demo folder run demo.py:
python demo.py
```

Each example prints P(TDE) after 1, 5, 10, 25... observations and opens a plot of the light curve above the
evolving probability. Use your own data with `python demo.py my_lightcurve.csv`
(columns `time,flux,flux_err,band`, bands `u g r i z y`). Add `--no-show` to save the figures without windows.
