### About
This is a game built for a neural network to play. A small circle (yellow) hunts for food particles (random colours) while avoiding killer particles (red). When the circle eats food it grows in size. And when it gets hit by a killer particle the size reduces.  
The neural network updates in real time and is supposed to get better at taking food particles while avoiding killer particles. I'm not sure the network learns very well... But looking at the circle you can see some behaviour that does not seem purely mechanical. Nevertheless I find it pleasing to look at.

### Run game
Setup a virtual environment and install the requirements
```python
$ pip install -r requirements.txt
```
Have a look in ```dashboard.py``` to set preferred parameters. You can choose to play yourself with the keyboard arrows by setting ```use_network``` to False.    
Run the game
```python
$ python dashboard.py
```

### Tune parameters
* You can tune parameters in the script dashboard.py
* If you want less hidden layers you can set a layer to 0.
* If you want to add more layers you have to modify the script accordingly.
* If you want to play yourself, set "use_network = False"

### NETWORK COMMENTS
It is assumed that the input a is an (n, 1) Numpy ndarray, not a (n,) vector. Here, n is the number of inputs to the network. If you try to use an (n,) vector as input you'll get strange results. Although using an (n,) vector appears the more natural choice, using an (n, 1) ndarray makes it particularly easy to modify the code to feedforward multiple inputs at once, and that is sometimes convenient.

## TODO
* Fix right-down preference(?)
* Run without rendering and print scores. Set round limit.
* Try network2,3

* Writing biases completely. Now thwy look like this:

    array(
        [
            [-0.54649949, -0.39720426,  0.24577855, ..., -0.56273681, -0.47515386, -1.43222448],
            [-1.07664751, -1.13604123, -0.35084798, ..., -1.0728256, -0.48466417, -2.09601431],
            [ 0.44930685, -0.13109731,  0.31155824, ...,  0.1682159, -0.50123441,  1.18888995],
            ...,
            [ 0.66755773, -0.21501595, -0.1728103 , ...,  1.15272542, -0.68289869,  0.41282875],
            [-0.28342745,  1.16772921,  0.10083753, ..., -0.69538828, 0.96084691,  0.13346749],
            [-1.95695393, -0.42648079, -1.32641063, ...,  0.32578252, -0.18378007,  0.57765803]
        ],
        shape=(32, 32)
    ),

* Run with a saved network
* Create neuroevolutional algoritgh that runs off render
