class output2hidden(object):
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_weight = np.random.randn(n_hidden, n_input + 1)
        self.output_weight = np.random.randn(n_output, n_hidden + 1)
        self.recurr_weight = np.random.randn(n_hidden, n_output + 1)
    ###中略###
    @classmethod
    def forward(self, x, y):
        r = self.recurr_weight.dot(np.hstack((1.0, y)))
        h = self.sigmoid(self.hidden_weight.dot(np.hstack((1.0, x))) + r)
        y= self.tanh(self.output_weight.dot(np.hstack((1.0, h))))
        return (h, y)

    def forward_seq_test(self, X):
        y = np.zeros(self.n_output)
        hs, ys = ([], [])
        for x in X:
            h, y = self.forward(x, y)
            hs.append(h)
            ys.append(y)
        return hs, ys

    def forward_seq_training(self, X, T):
        #'(あ)'
        T = np.concatenate((np.zeros(self.n_output).reshape(1,-1),T), axis=0)

        hs, ys = ([], [])
        for i in range(X.shape[0]):
            #h, y = self.forward(X[i], '(い)')
            h, y = self.forward(X[i], T[i])
            hs.append(h)
            ys.append(y)
        return hs, ys