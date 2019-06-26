import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#Hyperparameters
tree_levels=9
leafs_numb = 2**(tree_levels-1)
inner_numb = leafs_numb-1
k = 10
lmbd=0.1
alpha=0.01
num_epochs=41   #def:40
minibatch_size=64   #def:64


#Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images #shape(55.000,784)
train_labels = np.asarray(mnist.train.labels, dtype=np.int32) #shape(55.000,10)
cv_data = mnist.validation.images #shape(5.000,784)
cv_labels = np.asarray(mnist.validation.labels, dtype=np.int32) #shape(5.000,10)
eval_data = mnist.test.images #shape(10.000,784)
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32) #shape(10.000,10)



def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float64,[None,n_x],name="X")
    Y = tf.placeholder(tf.float64, [None, n_y], name="Y")
    V_prec = tf.placeholder(tf.float64, [None], name="V_prec")
    return X,Y,V_prec

def initialize_parameters(n_x):
    W = tf.get_variable("W", [n_x,inner_numb], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
    b = tf.get_variable("b", [1, inner_numb], initializer=tf.zeros_initializer(),dtype=tf.float64)
    phi = tf.get_variable("phi", [leafs_numb,k], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
    parameters = {"W": W,
                  "b": b,
                  "phi":phi}
    return parameters


#Compute the cross entropy leafs-samples 2D tensor. A tensor element i-j represents the cross entropy between softmax(leaf i) and sample j.
        #LEAFS_CE_MATRIX:
        #[Loss(L0,y0),Loss(L0,y1),...,Loss(L0,ym)]
        #[Loss(L1,y0),Loss(L1,y1),...,Loss(L1,ym)]
        #                  ...
        #[Loss(L_NL,y0),Loss(L_NL,y1),...,Loss(L_NL,ym)]  *NL=number of leafs
def compute_leafs_cross_entropy_matrix(phi,Y):
    m=tf.shape(Y)[0]
    ta = tf.TensorArray(dtype=tf.float64, size=leafs_numb)
    init_state = (0, ta)
    condition = lambda i, _: i < leafs_numb
    # tf.tile because each leafs is a "bigot" that always produce the same distribution
    body = lambda i, ta: (i + 1, ta.write(i,tf.nn.softmax_cross_entropy_with_logits(logits=tf.tile(tf.reshape(phi[i,:],[1,k]),[m,1]),
                                                                                  labels=Y)))
    n, ta_final = tf.while_loop(condition, body, init_state)
    leafs_ce_matrix = ta_final.stack()
    return leafs_ce_matrix


#Compute the probability of a tree node respect to all samples with activations in A.
        #P_FINAL:
        # P^index_node(X1),P^index_node(X2),...,P^index_node(Xm)
def node_probability(index_node,A):
    p=tf.ones([tf.shape(A)[0]],dtype=tf.float64)
    init_state=(index_node,p)
    condition = lambda index_node,_:index_node-1 >= 0
    def body(index_node,p):
        father_index=tf.floor_div(index_node-1,2)
        return(father_index,p*tf.cond(tf.equal((index_node - 1) % 2, 0),
                                      lambda: 1.0 - A[:,father_index],
                                      lambda: A[:,father_index]))
    n,p_final=tf.while_loop(condition,body,init_state)
    return p_final

#Compute the leafs probability matrix with respect to all m samples with activations in A.
    #LEAFS_PROB_MATRIX:
    # [P^0(X0),P^0(X1),...,P^0(Xm)]
    # [P^1(X0),P^1(X1),...,P^1(Xm)]
    #            ...
    #[P^NL(X0),P^NL(X1),...,P^NL(Xm)]
def compute_leafs_prob_matrix(A):
    ta = tf.TensorArray(dtype=tf.float64, size=leafs_numb)
    init_state = (0, ta)
    condition = lambda i, _: i < leafs_numb
    body = lambda i,ta: (i+1,ta.write(i,node_probability(leafs_numb-1+i,A)))
    n,ta_final=tf.while_loop(condition,body,init_state)
    leafs_prob_matrix=ta_final.stack()
    return leafs_prob_matrix


#Compute the inner nodes probability matrix with respect to all m samples with activations in A.
    #Tale matrice verra' utilizzata ai fini della regolarizzazione
    # [P^1(X1),P^1(X2),...,P^1(Xm)]
    # [P^2(X1),P^1(X2),...,P^1(Xm)]
    #            ...
    #[P^NI(X1),P^NI(X2),...,P^NI(Xm)]
def compute_inner_prob_matrix(A):
    ta = tf.TensorArray(dtype=tf.float64, size=inner_numb)
    init_state = (0, ta)
    condition = lambda i, _: i < inner_numb
    body = lambda i, ta: (i + 1, ta.write(i, node_probability(i, A)))
    n, ta_final = tf.while_loop(condition, body, init_state)
    inner_prob_matrix = ta_final.stack()
    return inner_prob_matrix

def compute_regularization(A,inner_prob_matrix,V_prec):
    #tensorarray for regularization terms of inner nodes
    ta = tf.TensorArray(dtype=tf.float64, size=inner_numb)
    # tensorarray for maintaining moving averages
    ema = tf.TensorArray(dtype=tf.float64, size=inner_numb)
    init_state = (0,ta,ema)
    condition = lambda i,_,_2: i < inner_numb
    def body(i,ta,ema):
        depth = tf.floor(tf.log(tf.cast(i,tf.float64) + 1)/tf.log(tf.constant(2,dtype=tf.float64)))
        # decay is exponentially proportional to the depth of the inner node and is used for the time windows
        # of the exponential decaying moving average as described in the paper.
        decay = 1. - tf.exp(-depth)
        a_i = tf.truediv(tf.tensordot(inner_prob_matrix[i, :], A[:, i], axes=1), tf.reduce_sum(inner_prob_matrix[i, :]))
        w_i=decay*V_prec[i]+(1.-decay)*a_i
        r_i = tf.reshape( -lmbd * (2 ** (-depth)) * (0.5 * tf.log(w_i) + 0.5 * tf.log(1.0 - w_i)), [1])
        return (i+1,ta.write(i,r_i),ema.write(i,w_i))
    n,ta_final,ema_final=tf.while_loop(condition,body,init_state)
    regularization=tf.reduce_sum(ta_final.stack())
    V_next=tf.reshape(ema_final.stack(),[inner_numb])
    return regularization,V_next


def forward_propagation(X, parameters):
    W = parameters['W']
    b = parameters['b']
    Z= tf.add(tf.matmul(X,W),b)
    A=tf.sigmoid(Z) #activations of inner nodes
    return A


def compute_cost(A,phi,Y,V_prec):
    leafs_ce_matrix=compute_leafs_cross_entropy_matrix(phi,Y)
    leafs_prob_matrix=compute_leafs_prob_matrix(A)
    inner_prob_matrix=compute_inner_prob_matrix(A)
    cost_wr=tf.reduce_mean(tf.reduce_sum(tf.multiply(leafs_ce_matrix,leafs_prob_matrix),0)) #cost without reg
    reg,V_next=compute_regularization(A,inner_prob_matrix,V_prec)
    cost=cost_wr+reg
    return cost,V_next


def compute_accuracy(parameters,X,Y):
    phi=parameters['phi']
    W=parameters['W']
    b=parameters['b']
    A = np.dot(X, W) + b
    m=X.shape[0]
    predictions = np.zeros(m)
    for i in range(m):
        index=0
        while index<A.shape[1]:
            if(A[i][index]>0.5):
                index=2*index+2
            else:
                index=2*index+1
        leaf_index=index-(leafs_numb-1)
        predictions[i]=(np.equal(np.argmax(phi[leaf_index]),np.argmax(Y[i]))).astype(float)
    accuracy=np.mean(predictions)
    return accuracy

def print_leafs_distributions(phi):
    for i in range(leafs_numb):
        print("Leaf ",i," class:",np.argmax(phi[i]))

def add_weights_to_summary(parameters):
    W=parameters['W']
    start_index=2**(tree_levels-2)-1
    end_index=start_index+2**(tree_levels-2)
    for i in range(start_index,end_index):
        weight_image = tf.reshape(W[:, i], [-1, 28, 28, 1])
        tf.summary.image("WEIGHT"+str(i),weight_image)

#---------------
#tf.set_random_seed(1)
(m,n_x)= train_data.shape
n_y= train_labels.shape[1]
costs=[]
X,Y,V_prec=create_placeholders(n_x,n_y)
parameters=initialize_parameters(n_x)
A=forward_propagation(X,parameters)
cost,V_next=compute_cost(A,parameters['phi'],Y,V_prec)
optimizer=tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost,var_list=parameters)
#For tensorboard summary
summary_cost = tf.placeholder(tf.float64, shape=(), name="cost")
summary_training_accuracy = tf.placeholder(tf.float64, shape=(), name="tr_acc")
summary_test_accuracy = tf.placeholder(tf.float64, shape=(), name="test_acc")
tf.summary.scalar("COST",summary_cost)
tf.summary.scalar("TRAINING ACCURACY",summary_training_accuracy)
tf.summary.scalar("TEST ACCURACY",summary_test_accuracy)
add_weights_to_summary(parameters)
merged_summary_op = tf.summary.merge_all()
#---------------

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as session:
    writer = tf.summary.FileWriter("logs/test3", session.graph)
    # V_next_value is used for the exponential moving average of a_i across various minibatches in training phase
    V_next_value=np.zeros(inner_numb)
    session.run(init)
    # saver.restore(session, "best_ckpt/model.ckpt-21")
    # par = session.run(parameters)
    # training_accuracy = compute_accuracy(par, train_data, train_labels)
    # test_accuracy = compute_accuracy(par, eval_data, eval_labels)
    # print("Accuracy on training set:", training_accuracy)
    # print("Accuracy on test set:", test_accuracy)
    for epoch in range(num_epochs):
        epoch_cost=0
        num_minibatches=int(m/minibatch_size)
        for minibatch in range(num_minibatches):
            minibatch_x, minibatch_y = mnist.train.next_batch(minibatch_size, shuffle=True)
            vn,_,minibatch_cost=session.run([V_next,optimizer,cost],feed_dict={X:minibatch_x,Y:minibatch_y,V_prec:V_next_value})
            epoch_cost+=minibatch_cost/num_minibatches
            V_next_value=vn
            #print("Cost minibatch %i: %f" % (minibatch, minibatch_cost))
        print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 5 == 0:
            save_path = saver.save(session,"tmp/test3/model.ckpt",global_step=epoch,write_meta_graph=False)
        par=session.run(parameters)
        training_accuracy=compute_accuracy(par,train_data,train_labels)
        test_accuracy=compute_accuracy(par, eval_data, eval_labels)
        print("Accuracy on training set:",training_accuracy)
        print("Accuracy on test set:", test_accuracy)
        summary = session.run(merged_summary_op,feed_dict={summary_cost:epoch_cost,
                                                           summary_training_accuracy:training_accuracy,
                                                           summary_test_accuracy:test_accuracy})
        writer.add_summary(summary, epoch)
    phi=session.run(parameters['phi'])
    print_leafs_distributions(phi)