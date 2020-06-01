# ML-NNet
Machine Learning - Neural Network Application

*Python*

This application was created to allow for neural networks to be applied within microservices using REST API endpoints.

# Installation

## Manual

To install the client download the github repo and then run:

```sh
python server.py
```

## Docker

To run from docker issue:

```sh
docker run -d -p 9000:9000 randysimpson/ml-nnet:latest
```

# REST API Endpoints

* POST `/api/v1/setup`

* POST `/api/v1/train`

* GET `/api/v1/use`

# Jupyter Notebook Example

[Neural Network](https://nbviewer.jupyter.org/url/raw.githubusercontent.com/randysimpson/ml-nnet/master/notebooks/Neural%20Network.ipynb)

# Curl Example

1. Setup the Neural Network 1 input layer, 1 output layer and 10 hidden layer network: `curl -i -d '{"xShape":1,"hiddenLayers": [10],"tShape":1,"nnType":"nn"}' -H 'Content-Type: application/json' -X POST http://localhost:9000/api/v1/setup`:

    ```sh
    root@ubuntu:$ curl -i -d '{"xShape":1,"hiddenLayers": [10],"tShape":1,"nnType":"nn"}' -H 'Content-Type: application/json' -X POST http://localhost:9000/api/v1/setup
    HTTP/1.0 200 OK
    Server: SimpleHTTP/0.6 Python/3.8.3
    Date: Sun, 31 May 2020 23:34:26 GMT
    Content-type: application/json

    {"status": "Initialized"}
    ```

2. Train the network `curl -i -d '{"epochs":2000,"method":"adam","learningRate":0.01,"x":[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],"t":[[2],[4],[8],[16],[32],[64],[128],[256],[512],[1024]]}' -H 'Content-Type: application/json' -X POST http://localhost:9000/api/v1/train`:

    ```sh
    root@ubuntu:$ curl -i -d '{"epochs":2000,"method":"adam","learningRate":0.01,"x":[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],"t":4],[128],[256],[512],[1024]]}' -H 'Content-Type: application/json' -X POST http://localhost:9000/api/v1/train
    HTTP/1.0 200 OK
    Server: SimpleHTTP/0.6 Python/3.8.3
    Date: Sun, 31 May 2020 23:35:15 GMT
    Content-type: application/json

    {"status": "Training"}
    ```

3. Wait for network to train, check status by using `curl -i -H 'Content-Type: application/json' -X GET http://localhost:9000/api/v1/status`:

    **Not Ready:**
    ```sh
    root@ubuntu:$ curl -i -H 'Content-Type: application/json' -X GET http://localhost:9000/api/v1/status
    HTTP/1.0 200 OK
    Server: SimpleHTTP/0.6 Python/3.8.3
    Date: Sun, 31 May 2020 23:27:20 GMT
    Content-type: application/json

    {"status": "Training", "nnType": "nn"}
    ```

    **Ready:**
    ```sh
    root@ubuntu:$ curl -i -H 'Content-Type: application/json' -X GET http://localhost:9000/api/v1/status
    HTTP/1.0 200 OK
    Server: SimpleHTTP/0.6 Python/3.8.3
    Date: Sun, 31 May 2020 23:35:42 GMT
    Content-type: application/json

    {"status": "Ready", "nnType": "nn"}
    ```

3. Use the trained neural network `curl -i -d '{"x":[[1],[3],[4],[5],[8],[20],[40]]}' -H 'Content-Type: application/json' -X GET http://localhost:9000/api/v1/use`:

    ```sh
    root@ubuntu:$ curl -i -d '{"x":[[1],[3],[4],[5],[8],[20],[40]]}' -H 'Content-Type: application/json' -X GET http://localhost:9000/api/v1/use
    HTTP/1.0 200 OK
    Server: SimpleHTTP/0.6 Python/3.8.3
    Date: Sun, 31 May 2020 23:38:27 GMT
    Content-type: application/json

    {"y": [[2.1799047309673085], [7.820670690549008], [16.511590610467437], [32.31600243401266], [254.9923804175499], [2665.2128328650647], [2663.631608763692]]}
    ```

# Licence

MIT License

Copyright (©) 2020 Randy Simpson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.