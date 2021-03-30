import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 入力層、隠れ層、出力層のノード数を設定する。
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 重みの初期化
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: self.activation_functionには、実装したシグモイド関数を設定します。 ####
        #
        # Note: in Python, 以下のように関数をラムダ式で定義することができます。

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # 0をあなたのシグモイド計算に置き換えてください。.
        
        ### 上のラムダコードが馴染みのないものであれば
        # 以下の3行をアンコメントアウトして、代わりに自分の実装を置くことができます。
        # 実装してください。
        #
        #def sigmoid(x):
        #    return 0  # 0をあなたのシグモイド計算に置き換えてください。
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' 特徴量とターゲットのバッチでネットワークを訓練します。 
        
            Arguments
            ---------
            
            features: 二次元配列, 各行が一つの記録データ, 各列が特徴量
            targets: targetsの1次元配列
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # 順伝播
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # 誤差逆伝播
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' ここに順伝播を実装
         
            Arguments
            ---------
            X: 特徴量バッチ
        '''
        #### ここに順伝播を実装 ####
        ### Forward pass ###
        #n_X = X.reshape((1,-1))

        #assert X.shape[1] == self.input_nodes
        # TODO: 隠れ層 - これらの値をあなたの計算で置き換えてください.
        inputs = np.array(X, ndmin=2).T
        inputs = inputs.reshape((1, -1))
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden) # 隠れ層への信号
        hidden_outputs = self.activation_function(hidden_inputs) # 隠れ層からの信号

        # TODO: 出力層 - これらの値をあなたの計算で置き換えてください..
        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # 最終出力層への信号
        final_outputs = final_inputs # 最終出力層からの信号
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' 誤差逆伝播の実装
         
            Arguments
            ---------
            final_outputs: 順伝播からの出力
            y: target (i.e. label) batch
            delta_weights_i_h: 入力層から隠れ層への重みの変化
            delta_weights_h_o: 隠れ層から出力層への重みの変化
        '''
        #### 誤差逆伝播をここに実装する ####
        ### Backward pass ###
        n_X = np.array(X, ndmin=2)
        n_X = n_X.reshape((1, -1))
        n_y = np.array(y, ndmin=2)
        n_y = n_y.reshape((1, -1))

        # TODO: 出力エラー - この値をあなたの計算値に置き換えてください。
        error = final_outputs - n_y # 出力層の誤差は、望ましい目標と実際の出力の差です。
        
        # TODO: 誤差逆伝播された誤差項 - これらの値をあなたの計算に置き換えてください。
        output_error_term =  error

        # TODO: 隠れ層の誤差への寄与を計算する
        hidden_output_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_input_error_term = hidden_output_error_term * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(n_X.T, hidden_input_error_term)
        # Weight step (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' 勾配降下ステップでの重みの更新
         
            Arguments
            ---------
            delta_weights_i_h: 入力層から隠れ層への重みの変化
            delta_weights_h_o: 隠れ層から出力層への重みの変化
            n_records: number of records
        '''
        #self.weights_hidden_to_output += -self.lr * np.sum(delta_weights_h_o, 0) / n_records # 勾配降下法による隠れた出力の重みの更新
        self.weights_hidden_to_output += -self.lr * delta_weights_h_o / n_records # 勾配降下法による隠れた出力の重みの更新
        #self.weights_input_to_hidden += -self.lr * np.sum(delta_weights_i_h, 0) / n_records # 勾配降下法による入力から隠れ層への重みの更新
        self.weights_input_to_hidden += -self.lr * delta_weights_i_h / n_records # 勾配降下法による入力から隠れ層への重みの更新

    def run(self, features):
        ''' 入力された特徴量を使って、準伝播を実行する。
        
            Arguments
            ---------
            features: 特徴量の1次元配列
        '''
        
        #### 順伝播をここに実装する ####
        # TODO: 隠れ層 - これらの値を適切な計算に置き換えてください。
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # 隠れ層への信号
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: 出力層 - これらの値を適切な計算値に置き換えてください。
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # 最終出力層への信号
        final_outputs = final_inputs# 最終出力層からの信号
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 4000
learning_rate = 0.5
hidden_nodes = 15
output_nodes = 1