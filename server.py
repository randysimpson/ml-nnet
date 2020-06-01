from http.server import BaseHTTPRequestHandler,HTTPServer, SimpleHTTPRequestHandler
import json
import numpy as np
import neuralnetwork as nn
import time
import threading

class GetHandler(SimpleHTTPRequestHandler):
    stats = {}
    nnStatus = {}

    def confusion_matrix(self, Ys, Ts):
        table = []
        class_names = []
        for true_class in np.unique(Ts):
            class_names.append(str(true_class))
            row = []
            for predicted_class in np.unique(Ts):
                row.append(100 * np.mean(Ys[Ts == true_class] == predicted_class))
            table.append(row)
        return table

    def generate_k_fold_cross_validation_sets(self, X, T, n_folds, shuffle=True):

        if shuffle:
            # Randomly order X and T
            randorder = np.arange(X.shape[0])
            np.random.shuffle(randorder)
            X = X[randorder, :]
            T = T[randorder, :]

        # Partition X and T into folds
        n_samples = X.shape[0]
        n_per_fold = round(n_samples / n_folds)
        n_last_fold = n_samples - n_per_fold * (n_folds - 1)

        folds = []
        start = 0
        for foldi in range(n_folds-1):
            folds.append( (X[start:start + n_per_fold, :], T[start:start + n_per_fold, :]) )
            start += n_per_fold
        folds.append( (X[start:, :], T[start:, :]) )

        # Yield k(k-1) assignments of Xtrain, Train, Xvalidate, Tvalidate, Xtest, Ttest

        for validation_i in range(n_folds):
            for test_i in range(n_folds):
                if test_i == validation_i:
                    continue

                train_i = np.setdiff1d(range(n_folds), [validation_i, test_i])

                Xvalidate, Tvalidate = folds[validation_i]
                Xtest, Ttest = folds[test_i]
                if len(train_i) > 1:
                    Xtrain = np.vstack([folds[i][0] for i in train_i])
                    Ttrain = np.vstack([folds[i][1] for i in train_i])
                else:
                    Xtrain, Ttrain = folds[train_i[0]]

                yield Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest

    def percent_correct(self, Predicted, Target):
        return 100 * np.mean(Predicted == Target)

    def do_GET(self):
        if self.path == '/api/v1/use':
            content_len = int(self.headers.get('Content-Length'))
            response_body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(response_body)

            rtnData = {}
            if self.nnStatus["status"] == "Ready":
                x = np.array(data["x"])
                Y = self.nnStatus["nnet"].use(x)

                rtnData["y"] = Y.tolist()

                if "t" in data:
                    rtnData["confusionMatrix"] = self.confusion_matrix(Y, np.array(data["t"]))

                self.send_response(200)
            else:
                rtnData["status"] = self.nnStatus["status"]

                self.send_response(400)
            
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            _data = json.dumps(rtnData)

            self.wfile.write(bytes(_data, 'utf8'))
            return
        elif self.path == '/api/v1/status':
            rtnData = {}
            if len(self.stats) > 0:
                rtnData["stats"] = self.stats
            rtnData["status"] = self.nnStatus["status"]
            if "nnType" in self.nnStatus:
                rtnData["nnType"] = self.nnStatus["nnType"]

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            print(rtnData)
            _data = json.dumps(rtnData)

            self.wfile.write(bytes(_data, 'utf8'))
            return
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            return

            #rtnData["errorTrace"] = self.nnet.error_trace

    def do_POST(self):
        if self.path == '/api/v1/setup':
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(post_body)

            xShape = data["xShape"]
            hiddenLayers = data["hiddenLayers"]
            tShape = data["tShape"]
            nnType = data["nnType"] #classifier, recuring, etc

            self.nnStatus["nnType"] = nnType
            if nnType == "classifier":
                self.nnStatus["nnet"] = nn.NeuralNetworkClassifier(xShape, hiddenLayers, tShape)
            else:
                self.nnStatus["nnet"] = nn.NeuralNetwork(xShape, hiddenLayers, tShape)
            
            self.nnStatus["status"] = "Initialized"

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            rtnData = {}
            rtnData["status"] = self.nnStatus["status"]
            _data = json.dumps(rtnData)

            self.wfile.write(bytes(_data, 'utf8'))
            return
        elif self.path == '/api/v1/train':
            def trainNet(data):
                epochs = data["epochs"]
                learningRate = data["learningRate"]
                method = data["method"] #adam or sgd
                x = np.array(data["x"])
                t = np.array(data["t"])

                rtnData["status"] = "Training"
                self.nnStatus["status"] = "Training"
                self.start = time.time()

                self.nnStatus["nnet"].train(x, t, epochs, learningRate, method=method, verbose=False)
                self.elapsed = (time.time() - self.start)

                self.nnStatus["status"] = "Ready"

            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(post_body)

            rtnData = {}

            print(self.nnStatus["status"])
            if self.nnStatus["status"] in ["Initialized", "Ready"]:
                self.send_response(200)
                self.nnStatus["status"] = "Training"
                thread = threading.Thread(target=trainNet, kwargs={'data': data})
                thread.start()
            else:
                self.send_response(400)
                rtnData["message"] = "Not ready for training, must be in initialized or ready status."

            rtnData["status"] = self.nnStatus["status"]
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            post_data = json.dumps(rtnData)

            self.wfile.write(bytes(post_data, 'utf8'))
            return
        elif self.path == '/api/v1/full':
            def trainFold(data):
                self.stats["train"] = {}
                self.stats["train"]["x"] = []
                self.stats["train"]["t"] = []
                self.stats["train"]["y"] = []
                self.stats["train"]["correct"] = []
                self.stats["validate"] = {}
                self.stats["validate"]["x"] = []
                self.stats["validate"]["t"] = []
                self.stats["validate"]["y"] = []
                self.stats["validate"]["correct"] = []
                self.stats["test"] = {}
                self.stats["test"]["x"] = []
                self.stats["test"]["t"] = []
                self.stats["test"]["y"] = []
                self.stats["test"]["correct"] = []
                for Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest in self.generate_k_fold_cross_validation_sets(X, T, folds, shuffle):
                    self.nnStatus["nnet"].train(Xtrain, Ttrain, epochs, learningRate, method=method, verbose=False)
                    Ytrain = self.nnStatus["nnet"].use(Xtrain)
                    self.stats["train"]["correct"].append(self.percent_correct(Ytrain[0], Ttrain))
                    if self.nnStatus["nnType"] == "classifier":
                        Ytrain = [Ytrain[0].tolist(), Ytrain[1].tolist()]
                    else:
                        Ytrain = Ytrain.tolist()
                    self.stats["train"]["x"].append(Xtrain.tolist())
                    self.stats["train"]["y"].append(Ytrain)
                    self.stats["train"]["t"].append(Ttrain.tolist())
                    Yvalidate = self.nnStatus["nnet"].use(Xvalidate)
                    self.stats["validate"]["correct"].append(self.percent_correct(Yvalidate[0], Tvalidate))
                    if self.nnStatus["nnType"] == "classifier":
                        Yvalidate = [Yvalidate[0].tolist(), Yvalidate[1].tolist()]
                    else:
                        Yvalidate = Yvalidate.tolist()
                    self.stats["validate"]["x"].append(Xvalidate.tolist())
                    self.stats["validate"]["y"].append(Yvalidate)
                    self.stats["validate"]["t"].append(Tvalidate.tolist())
                    Ytest = self.nnStatus["nnet"].use(Xtest)
                    self.stats["test"]["correct"].append(self.percent_correct(Ytest[0], Ttest))
                    if self.nnStatus["nnType"] == "classifier":
                        Ytest = [Ytest[0].tolist(), Ytest[1].tolist()]
                    else:
                        Ytest = Ytest.tolist()
                    self.stats["test"]["x"].append(Xtest.tolist())
                    self.stats["test"]["y"].append(Ytest)
                    self.stats["test"]["t"].append(Ttest.tolist())
                self.nnStatus["status"] = "Ready"
                print(self.nnStatus["status"])

            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(post_body)

            hiddenLayers = data["hiddenLayers"]
            nnType = data["nnType"] #classifier, recuring, etc
            epochs = int(data["epochs"])
            learningRate = float(data["learningRate"])
            method = data["method"] #adam or sgd
            dataSet = np.array(data["data"])
            tColumnCount = int(data["tColumnCount"])
            folds = int(data["folds"])
            shuffle = data["shuffle"]

            X = dataSet[:, :-tColumnCount]
            T = dataSet[:, -tColumnCount:]

            self.nnStatus["nnType"] = nnType
            if nnType == "classifier":
                self.nnStatus["nnet"] = nn.NeuralNetworkClassifier(X.shape[1], hiddenLayers, len(np.unique(T)))
            else:
                self.nnStatus["nnet"] = nn.NeuralNetwork(X.shape[1], hiddenLayers, T.shape[1])

            rtnData = {}
            self.nnStatus["status"] = "Training"
            rtnData["status"] = "Training"

            thread = threading.Thread(target=trainFold, kwargs={'data': data})
            thread.start()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            post_data = json.dumps(rtnData)

            self.wfile.write(bytes(post_data, 'utf8'))
            return

Handler=GetHandler
print("Starting on port 9000")
httpd=HTTPServer(("0.0.0.0", 9000), Handler)
httpd.serve_forever()