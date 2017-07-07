
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

# create a constant op
# This op is added as a node to the default graph
# 문자열이 들어있는 하나의 텐서플로우 constant 노드가 만들어 진 것이다
hello = tf.constant("Hello, TensorFlow!")

# start a TF session 
# constant를 실행하기 위해서는 session이란 것을 만들어야 한다.
sess = tf.Session()

# run the op and get result
# hello 라는 노드를 실행시킨다
# 출력시 앞에 뜨는 'b' : bytestream @python3
print(sess.run(hello))


# In[3]:

# STEP1 : Build graph(tensors) using TF operations

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2) # == node3 = node1+node2
# node3은 더하기 노드가 될 것이다.


# In[4]:

print("node1 : ",node1, "node2 : ", node2)
print("node3 : ", node3)

# print의 결과값 : 그냥 단순히 각 노드가 텐서임을 보여준다. 결과값 안나옴


# In[5]:

# STEP2 : feed data and run graph(operation) -> sess.run(op)
# STEP3 : update variables in the graph(and return values)

sess = tf.Session()

# 따라서 결과 값을 나오게 하고 싶다면 세션을 만들고, 그 세션을 run해서 그래프의 노드를 실행시킨다
print("sess.run(node1, node2) : ", sess.run([node1,node2]))
print("sess.run(node3) : ", sess.run(node3)) # 더하기 노드 실행시키자


# In[7]:

# node를 "placeholder" 라는 특별한 노드로 만들어준다.
# placeholder : 프로그램 실행 중 값을 변경할 수 있는 심볼릭 변수. double a / int b 같은거인듯..
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b # + probides a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

