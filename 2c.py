
def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)


filename = '/content/drive/My Drive/Colab Notebooks/2a.xls'
all_data = pd.read_excel(filename, usecols=(1, 2, 3, 4, 5, 6, 7))
data = dict()

data['x'] = all_data['temperature'][:10]
data['z'] = all_data['humidity'][:10]
data['k'] = all_data['lowcost_pm2_5'][:10]

values = {'x' :0 ,'z' :0 ,'k' :0}
data['y'] = all_data['reference_pm2_5'][:10]

nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)

data['x'] = data['x'][indices]
data['y'] = data['y'][indices]
data['z'] = data['z'][indices]
data['k'] = data['k'][indices]

data['y'] = data['y'].str.replace(',' ,'.').astype(float)

data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['z'] = (data['z'] - np.mean(data['z'], axis=0)) / np.std(data['z'], axis=0)
data['k'] = (data['k'] - np.mean(data['k'], axis=0)) / np.std(data['k'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

nb_features = 3

data['x'] = create_feature_matrix(data['x'], nb_features)
data['z'] = create_feature_matrix(data['z'], nb_features)
data['k'] = create_feature_matrix(data['k'], nb_features)

plt.scatter(data['x'][: ,0] ,data['y'])
plt.scatter(data['z'][: ,0] ,data['y'])
plt.scatter(data['k'][: ,0] ,data['y'])

X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32, name="X")
Y = tf.placeholder(shape=(None), dtype=tf.float32, name="Y")
w = tf.Variable(tf.zeros(nb_features) ,name="w")
bias = tf.Variable(0.0)

w_col = tf.reshape(w, (nb_features, 1) ,name="w_col")
hyp = tf.add(tf.matmul(X, w_col), bias ,name="hyp")

Y_col = tf.reshape(Y, (-1, 1) ,name="Y_col")

lmbd_array = [0 ,0.001 ,0.01 ,0.1 ,1 ,10 ,100]

lmbd = tf.placeholder("float", None)
l2_reg = lmbd * tf.reduce_mean(tf.square(w))

mse = tf.reduce_mean(tf.square(hyp - Y_col), name="mse")
loss = tf.add(mse, l2_reg ,name="loss")


opt_op = tf.train.AdamOptimizer().minimize(loss)

def show(loss):
    w_val = sess.run(w)
    bias_val = sess.run(bias)
    xs = create_feature_matrix(np.linspace(-3, 4, 100), nb_features)
    hyp_val = sess.run(hyp, feed_dict={X: xs})
    plt.plot(xs[:, 0].tolist(), hyp_val.tolist())
    plt.xlim([-3 ,3])
    plt.ylim([-2 ,2])


ftrs = ['x' ,'z' ,'k']
loss_arr = []
final_loss = 0


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for ftr in ftrs:
        nb_epochs = 100
        for val in lmbd_array:
            for epoch in range(nb_epochs):

                epoch_loss = 0
                for sample in range(nb_samples):

                    feed = {X: data[ftr][sample].reshape((1, nb_features)),
                            Y: data['y'][sample],
                            lmbd: [val]}
                    _, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
                    epoch_loss += curr_loss

                epoch_loss /= nb_samples

            show(epoch_loss)
            final_loss = sess.run(loss, feed_dict={X: data[ftr], Y: data['y'] ,lmbd: [val]})
            loss_arr.append(final_loss)




plt.show()


xpoints = np.array(loss_arr)
ypoints = np.array(lmbd_array)

plt.plot(np.reshape(xpoints[:7] ,7) ,ypoints)
plt.plot(np.reshape(xpoints[7:14] ,7) ,ypoints)
plt.plot(np.reshape(xpoints[14:] ,7) ,ypoints)

plt.show()
