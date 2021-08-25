import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data():
    print("[#]Loading data")
    data = np.genfromtxt("datas/iris/iris.csv", delimiter=",")
    label = np.genfromtxt("datas/iris/target.csv", delimiter=",").astype(int)

    # L, W = data.shape
    print("[*]data shape:", data.shape)
    label = np.eye(len(np.unique(label)))[label]
    print("[*]label shape:", label.shape)

    # all_x = np.ones((L, W + 1))
    # all_x[:, 1:] = data
    # print("[*]x shape:", all_x.shape)
    # print(data)
    # print(all_x)
    # data = tf.convert_to_tensor(data)
    # label = tf.convert_to_tensor(label)
    return data, label


def forward(x, datas):
    h1 = tf.nn.sigmoid(tf.matmul(x, datas["w1"]) + datas["b1"])
    y = tf.matmul(h1, datas["w2"]) + datas["b2"]
    return y


# #########################DATAS################################
xs, ys = load_data()
x_sz = xs.shape[1]
y_sz = ys.shape[1]
h_sz = 32

datas = {}
X = tf.placeholder(tf.float32, shape=[None, x_sz])
Y = tf.placeholder(tf.int32, shape=[None, y_sz])
datas["w1"] = tf.Variable(tf.random_normal([x_sz, h_sz]), name="w1")
datas["b1"] = tf.Variable(tf.zeros([h_sz]), name="b1")
datas["w2"] = tf.Variable(tf.random_normal([h_sz, y_sz]), name="w2")
datas["b2"] = tf.Variable(tf.zeros([y_sz]), name="b2")

y_pred = forward(X, datas)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))

opt = tf.train.AdamOptimizer(0.05).minimize(cost)

maxEpochs = 10
minibitchSize = 10
cost_datas = []

with tf.Session() as sess:
    print("[#]start")
    sess.run(tf.initialize_all_variables())
    for epoch in range(maxEpochs):
        for i in range(int(ys.shape[0] // minibitchSize)):
            x1 = xs[i * minibitchSize:(i + 1) * minibitchSize]
            y1 = ys[i * minibitchSize:(i + 1) * minibitchSize]
            _, cost_data = sess.run([opt, cost], feed_dict={X: x1, Y: y1})
            cost_datas.append(cost_data)
        print("[+]Epoch:", epoch, "cost:", cost_datas[-1])
print("[!]Finish")

plt.plot([x for x in range(len(cost_datas))], cost_datas, label="cost_data")
plt.legend()
plt.show()
