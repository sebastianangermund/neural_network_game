OBS Run with python3.6

### Run game

```python
$ python dashboard.py
```

### Tune parameters
* You can tune parameters in the script dashboard.py.
* If you want less hidden layers you can set a layser to 0.
* If you want to add more layers you have to modify the script accordingly.

If you want to play yourself, set "use_network = False"

NETWORK COMMENTS

It is assumed that the input a is an (n, 1) Numpy ndarray, not a (n,) vector. Here, n is the number of inputs to the network. If you try to use an (n,) vector as input you'll get strange results. Although using an (n,) vector appears the more natural choice, using an (n, 1) ndarray makes it particularly easy to modify the code to feedforward multiple inputs at once, and that is sometimes convenient.
