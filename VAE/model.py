from utils import *
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools,time
from sklearn.model_selection import train_test_split

slim = tf.contrib.slim

tf.reset_default_graph()

class VAE(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """


    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.01, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        print('Initial Learning Rate:', self.learning_rate)

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]], name='input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.h_dim = (network_architecture["n_z"]) # had a float before
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)                         # a    = 1
        self.prior_mean = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)          # prior_mean  = 0
        self.prior_var = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +       # prior_var = 0.99 + 0.005 = 0.995
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )
        self.prior_logvar = tf.log(self.prior_var)
        self.means = []

        self._create_network()
        with tf.name_scope('cost'):
            self._create_loss_optimizer()

        init = tf.initialize_all_variables()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        """
        steps:
        1. initialize weights
        2. build recognition network
        3. build reconstruction network
        """
        n_z = self.network_architecture['n_z']
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        en1 = slim.layers.linear(self.x, self.network_architecture['n_hidden_recog_1'], scope='FC_en1')
        en1 = tf.nn.softplus(en1, name='softplus1')
        en2 = slim.layers.linear(en1,    self.network_architecture['n_hidden_recog_2'], scope='FC_en2')
        en2 = tf.nn.softplus(en2, name='softplus2')
        en2_do = slim.layers.dropout(en2, self.keep_prob, scope='en2_dropped')
        self.posterior_mean   = slim.layers.linear(en2_do, self.network_architecture['n_z'], scope='FC_mean')
        self.posterior_logvar = slim.layers.linear(en2_do, self.network_architecture['n_z'], scope='FC_logvar')
        self.posterior_mean   = slim.layers.batch_norm(self.posterior_mean, scope='BN_mean')
        self.posterior_logvar = slim.layers.batch_norm(self.posterior_logvar, scope='BN_logvar')
        
        with tf.name_scope('z_scope'):
            eps = tf.random_normal((self.batch_size, n_z), 0, 1,                            # take noise
                                   dtype=tf.float32)
            self.z = tf.add(self.posterior_mean,
                            tf.multiply(tf.sqrt(tf.exp(self.posterior_logvar)), eps))         # reparameterization z
            self.posterior_var = tf.exp(self.posterior_logvar) 

        self.p = slim.layers.softmax(self.z)
        p_do = slim.layers.dropout(self.p, self.keep_prob, scope='p_dropped')               # dropout(softmax(z))
        decoded = slim.layers.linear(p_do, n_hidden_gener_1, scope='FC_decoder')

        self.x_reconstr_mean = tf.nn.softmax(slim.layers.batch_norm(decoded, scope='BN_decoder'))                    # softmax(bn(50->1995))

        print(self.x_reconstr_mean)

    def _create_loss_optimizer(self):

        tensor = self.x * tf.log(self.x_reconstr_mean+1e-10)                                                   # prevent log(0)
        indices = [i for i in range(1,tensor.shape[1])] # exclude 'pad', include 'unk'
        result = tf.gather(tensor, indices, axis=1)
        NL = -tf.reduce_sum(result, 1)
        # NL = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean+1e-10), 1)     # cross entropy on categorical:- sum(ylog(p))

        var_division    = self.posterior_var  / self.prior_var
        diff            = self.posterior_mean - self.prior_mean
        diff_term       = diff * diff / self.prior_var
        logvar_division = self.prior_logvar - self.posterior_logvar
        KLD = 0.5 * (tf.reduce_sum(var_division + diff_term + logvar_division, 1) - self.h_dim )

        self.cost = tf.reduce_mean(NL + KLD)
 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99,epsilon=0.01).minimize(self.cost)

    def partial_fit(self, X):

        #if hasattr(self, 'decoder_weight'):
            #decoder_weight = self.decoder_weight
        #else:
        decoder_weight = [v for v in tf.global_variables() if v.name=='FC_decoder/weights:0'][0]
        opt, cost,emb,p = self.sess.run((self.optimizer, self.cost, decoder_weight, self.p),feed_dict={self.x: X,self.keep_prob: .8})
        # print(self.sess.run((self.p),feed_dict={self.x: X,self.keep_prob: .8}))
        return cost,emb,p

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run((self.cost),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return cost
    def topic_prop(self, X):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        theta_ = self.sess.run((self.z),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return theta_

def model_train(train_size,network_architecture, minibatches, val, learning_rate=0.01,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae = VAE(network_architecture,transfer_fct=tf.nn.softplus,
                learning_rate=learning_rate, batch_size=batch_size)
    writer = tf.summary.FileWriter('logs', tf.get_default_graph())
    emb=0
    # Training cycle
    best_val_ppl = 10000
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_size / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next(minibatches)
            # Fit training using batch data
            cost,emb,p = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / train_size * batch_size

            if np.isnan(avg_cost):
                print(epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()
        # record best val ppl
        val_ppl = cal_val_ppl(vae,val)
        best_val_ppl = min(best_val_ppl,val_ppl)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "avg train cost=", "{:.9f} approximated test PPL is: {:.9f}".format(avg_cost,val_ppl))
            
    return vae,emb,best_val_ppl

def search_best_params(data, max_vocab_size, learning_rate, batch_size, layer1, layer2, num_topics, epochs):
    #train,test split 
    train, test = train_test_split(data, test_size=0.2)
    #build vocab for train and index for train,test
    all_tokens = []
    for i in train:
        all_tokens += i
    token2id, id2token = build_vocab(all_tokens,max_vocab_size)

    x_train = token2index_dataset(train, token2id, id2token)
    x_train = np.array([np.array(document) for document in x_train]) 
    x_train = np.array([onehot(doc.astype('int'),max_vocab_size+2) for doc in x_train if np.sum(doc)!=0])
    train_size = x_train.shape[0]
    x_val = token2index_dataset(test, token2id, id2token)
    x_val = np.array([np.array(document) for document in x_val])
    x_val = np.array([onehot(doc.astype('int'),max_vocab_size+2) for doc in x_val if np.sum(doc)!=0])

    #collate batches
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=x_train.shape[1], # 1st layer decoder neurons
             n_input=x_train.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space

    minibatches = create_minibatch(x_train.astype('float32'),batch_size=batch_size)

    return model_train(train_size,network_architecture, minibatches,x_val,learning_rate,batch_size, training_epochs=epochs, display_step=5)
