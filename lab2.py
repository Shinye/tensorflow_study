
# coding: utf-8

# In[5]:

import tensorflow as tf


# In[7]:

### Graph build

# X and Y data
x_train = [1,2,3] # x 값을 집어넣으면
y_train = [1,2,3] # y 값이 출력된다.

# Placeholder를 사용하는 경우.
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# tensorflow가 실행되면 얘네가 학습하는 과정에서 자체적으로 변경시키는 값 : variable
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis : Wx+b
hypothesis = x_train * W + b


# In[8]:

# cost/lost function
# reduce_mean : 평균 내주는 거
cost = tf.reduce_mean(tf.square(hypothesis - y_train))


# In[9]:

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) 
# cost값을 minimize하기 위해 variable 값을 자동으로 조정한다.
# train은 그래프 내의 한 노드의 이름이 될 것.


# In[10]:

### Run/update graph and get results

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables(W,b) in the graph
# 텐서플로우에서 variable을 사용하기 위해서는 global_variables_initializer를 사용해준다.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train) 
    # 이 train노드를 실행시킨다. 
    # train노드를 실행시키면 이 노드와 연결된 노드들도 실행되기 때문에 cost, hypo~, W, b 노드 모두 실행된다.
    sess.run(tf.global_variables_initializer())
    
    # placeholder 사용경우
    # cost_val, W_val, b_val, _ = \
    # sess.run([cost,W,b,train],
    # feed_dict={X:[1,2,3], Y:[1,2,3]})
    
    if step%20 == 0: # 스텝을 스무번에 한번씩 보고 출력시키자.
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        # print(step, cost_val, W_val, b_val)


# In[ ]:



