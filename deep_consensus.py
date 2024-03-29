'''
Distributed Tensorflow 1.12.0 example of using data parallelism and the consensus worker communication scheme.
Trains a deep convolutional neural network on MNIST for 10 epochs on n_nodes machines using one parameter server.
The goal is to put minimum workload on the parameter server such that all computations are assigned to the workers.
The distribution model is the consensus model, based on which each worker communicates only with its neighbours based
on the topology provided as input (supported: 'full', 'ring', 'random'), averages their computed gradients and applies the
aggregation to its current parameter vector(s); then it proceeds to compute the next gradient and send it to its neighbors.

pc-01$ python example.py --job_name="ps" --task_index=0
pc-02$ python example.py --job_name="worker" --task_index=0
pc-03$ python example.py --job_name="worker" --task_index=1
pc-04$ python example.py --job_name="worker" --task_index=2
'''

from __future__ import print_function
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data
from topology import get_graph
import numpy as np
import os
import random as rn
import sys
import time
import tensorflow as tf
import worker

# Set all possible seeds
os.environ['PYTHONHASHSEED'] = '0'
SEED = 5
tf.set_random_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

# Raise error if trying to seed after graph construction
if len(tf.get_default_graph()._nodes_by_id.keys()) > 0:
    raise RuntimeError("Seeding is not supported after building part of the graph. "
                       "Please move set_seed to the beginning of your code.")
    
# Read arguments from input file
with open("input.txt", "r") as f:
    input_args = [line.split("=")[-1].rstrip("\n") for line in f]
    n_nodes, topology = input_args
    n_nodes = int(n_nodes)

# Cluster specification; one port for the PS (usually 2222) and as many ports as needed
# for the workers
parameter_servers = ["localhost:2222"]
workers = ["localhost:{}".format(i) for i in range(2223, 2223 + n_nodes)]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# Network Topology
topology = get_graph(n_nodes, topology)

# Input flags; define if it is a PS or a worker and its corresponding task number
tf.app.flags.DEFINE_string("job_name", "worker", "either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# Start a server for a specific task
server = tf.train.Server(cluster,
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# Config; this info can go in the input file;
# hardcoding was more convenient for testing
batch_size = 100
learning_rate = 0.005
training_epochs = 10
logs_path = "/tmp/mnist/1"
wdevs = [i for i in range(len(workers))]    # number of worker devices

# Load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, seed=SEED)


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Instantiate a worker object to hold its properties: ID, parameter
    # vector and gradients
    worker = worker.Worker(FLAGS.task_index)

    # Initialize the queues on the chief worker
    # The dimensions of the tensors that the queues will receive should be predefined 
    with tf.device("/job:worker/task:0/cpu:0"):
        token_queues = []
        dtypes = [tf.float32]*8
        q_shapes = [[5, 5, 1, 32],
                    [5, 5, 32, 64],
                    [7 * 7 * 64, 1024],
                    [1024, 10],
                    [32],
                    [64],
                    [1024],
                    [10]]

        # 1 outgoing queue for each worker's neighbour 
        # token_queues will be a n_workers-length list with n_workers length item
        # so essential a n_workers x n_workers matrix, which we can index
        # the diagonal is redundant obviously - no worker sends to itself
        for wdev in wdevs:
            this_wdev_queues = [tf.FIFOQueue(1,
                                 dtypes=dtypes,
                                 shapes=q_shapes,
                                 name="from_{}_to_{}/q".format(wdev, item),
                                 shared_name="from_{}_to_{}/q".format(wdev, item)) for item in wdevs]
            token_queues.append(this_wdev_queues)


    # Between-graph replication; start assigning individual jobs to individual workers
    full_device_name = "/job:worker/task:%d" % worker._id
    with tf.device(tf.train.replica_device_setter(
                worker_device=full_device_name,
                cluster=cluster)):

        # Placeholders for input tensors
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # Target: 10 classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")


        # Initialize weights and biases (parameter vectors)
        with tf.name_scope("weights"):
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=SEED))
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED))
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1, seed=SEED))
            W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1, seed=SEED))

        # Bias
        with tf.name_scope("biases"):
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))


        # Assign initialized variables to current worker
        worker.get_vars([W_conv1, W_conv2, W_fc1, W_fc2, b_conv1, b_conv2, b_fc1, b_fc2])

        # Initial enqueue list of operations which enqueue initial PVs to all queues
        init_enq_op = []
        for neighbour in topology[worker._id]:
            op = token_queues[worker._id][neighbour].enqueue(worker.vars_)
            init_enq_op.append(op)

        # Function for applying gradients to current PV
        # The result is multiplied by the learning_rate
        def grads_x_lr(grads, vars_, learning_rate):

            mul_grads = []
            for grad, var in zip(grads, vars_):
                mul_grad = tf.scalar_mul(learning_rate, grad)
                mul_grads.append(mul_grad)
            return mul_grads


        with tf.name_scope("softmax"):
            assign_ops = []
            for tensor1, tensor2 in zip((W_conv1, W_conv2, W_fc1, W_fc2, b_conv1, b_conv2, b_fc1, b_fc2), worker.vars_):
                assign_ops.append(tf.assign(tensor1, tensor2))
                
            # This line must be equivalent to the above assign_ops
            # but we need the assign_ops as a dependency below
            # W1, W2, b1, b2 = worker.vars_
            
            # Our deep CNN model, always subject to changes
            with tf.control_dependencies(assign_ops):
                x_image = tf.reshape(x, [-1, 28, 28, 1])
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, 0.8)
                # y is our prediction
                y = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="Prediction")

        # Specify cost function
        with tf.name_scope('cross_entropy'):
            # both should be equivalent
            #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))


        with tf.name_scope('compute_grads'):
            # compute and assign gradients to the current worker
            worker.get_grads(cross_entropy, tf.trainable_variables())

        # dequeue from neighbours 
        pvs_nbr = [token_queues[neighbour][worker._id].dequeue() for neighbour in topology[worker._id]]
        # stack the current PV to the neighbour ones
        pvs_nbr.append(worker.vars_)
        pvs_zipped = zip(*pvs_nbr)
        # average all the PVs
        mean_pvs = [tf.reduce_mean(item, axis=0) for item in pvs_zipped]

        # A print operation that outputs the maximum value of the W_conv1 vector across 2 axes
        # in order to check its value among different workers; 
        with tf.name_scope("print_operations2"), tf.control_dependencies([W_conv1]):
            max_var0 = tf.reduce_max(tf.reduce_max(W_conv1, reduction_indices=[1]), reduction_indices=[0])
            print_ops2 = [tf.print("Worker {} reduce max variable".format(worker._id), max_var0, output_stream=sys.stdout)]
            # better: could be replaced with the norm
            #w1_norm = tf.norm(W_conv1)
            #print_ops2 = [tf.print("Worker {} reduce max variable".format(worker._id), w1_norm, output_stream=sys.stdout)]

        # Now we deduct the grad 
        with tf.control_dependencies(print_ops2), tf.name_scope('apply_grads'):
        #with tf.name_scope('apply_grads'):
            mul_grads = grads_x_lr(worker.grads, worker.vars_, learning_rate)
            new_vars = [tf.subtract(mean_var, mul_grad) for mean_var, mul_grad in zip(mean_pvs, mul_grads)]

            # Update the worker's Parameter Vector
            #worker.vars_ = new_vars
            assign_ops2 = []
            for tensor1, tensor2 in zip(worker.vars_, new_vars):
                assign_ops2.append(tf.assign(tensor1, tensor2))

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Enqueue new PV in neighbours queues
        with tf.control_dependencies(assign_ops2):
            enq_ops = [token_queues[worker._id][neighbour].enqueue(worker.vars_) for neighbour in topology[worker._id]]

        # In case we need to run another print operation separately 
        # this could be train_op = enq_ops + [print_op]
        train_op = enq_ops

        # Configure session; make sure that parallelism is set to 1
        # for reproducible results
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0))
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
        
        begin_time = time.time()
        frequency = 100
        with sv.prepare_or_wait_for_session(server.target, config=session_conf) as sess:
            np.random.seed(SEED)
            tf.set_random_seed(SEED)
            rn.seed(SEED)
            
            # filenames for results
            f0_name = "consensus_results_loss_iters{}.txt".format(FLAGS.task_index)
            f1_name = "consensus_results_loss_time{}.txt".format(FLAGS.task_index)
            f2_name = "consensus_results_time_iters{}.txt".format(FLAGS.task_index)
            
            with open(f0_name,"w") as f0, open(f1_name,"w") as f1, open(f2_name,"w") as f2:
                start_time = time.time()
                # for the plots against iterations
                overall_count = 0 
                # first run the initial enqueue operation
                sess.run(init_enq_op) 
                
                for epoch in range(training_epochs):
                    # number of batches in one epoch
                    batch_count = int(mnist.train.num_examples/batch_size)
                    count = 0
                    print("Number of batches in dataset: ",batch_count)
                    
                    for i in range(batch_count):
                        print("Iteration NO: ", count)
                        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=False)

                        # perform the operations we defined earlier on batch
                        _, cost = sess.run([train_op, cross_entropy],
                                            feed_dict={x: batch_x, y_: batch_y})
                        count += 1
                        overall_count += 1
                        elapsed_time = time.time() - begin_time
                        
                        # write on output files
                        f0.write("{0:d}\t{1:.4f}\n".format(overall_count, cost))
                        f1.write("{0:.2f}\t{1:.4f}\n".format(elapsed_time, cost))
                        f2.write("{0:d}\t{1:.2f}\n".format(overall_count, elapsed_time))
                        
                        # Print every 100 iterations and in the end
                        if count % frequency == 0 or i+1 == batch_count:
                            elapsed_time = time.time() - start_time
                            start_time = time.time()
                            print(" Epoch: %2d," % (epoch+1),
                              " Batch: %3d of %3d," % (i+1, batch_count),
                              " Cost: %.4f," % cost,
                              " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                            count = 0

            print("Train-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            print("Final Cost: %.4f" % cost)

            sv.stop()


print("done")
