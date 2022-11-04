sequence_length = 3
input_size = 2
output_size1 = 16
output_size_N = 32
output_size_S = 16
output_size2 = 16
model = tf.keras.Sequential()
model.add(MIM_AE(output_size1,output_size_N, output_size_S, output_size2,return_sequences = True))
model.add(tf.keras.layers.Dense(2,name='decoder'))
def mse(y_true,y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
    
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=0.01))########################'tf.keras.losses.sparse_categorical_crossentropy'

data_train = data1[:560,:]
data_valid = data1[560:800,:]
data_test = data1[800:,:]
data_fault = data1[1000:1280,:]

data = data1[:560,:]
valid = data1[560:800,:]
mean = np.mean(data,axis=0)
std=np.std(data,axis=0)
data = preprocessing.scale(data)
data_batch = []
for i in range(len(data)-sequence_length+1):
    data_batch.append(data[i:i+sequence_length,:])
data_batch=np.array(data_batch)
data_batch=data_batch.reshape(len(data)-sequence_length+1,sequence_length,2)
x_train = tf.convert_to_tensor(data_batch[:,:,:])

valid = (valid-mean)/std
valid_batch = []
for i in range(len(valid)-sequence_length+1):
    valid_batch.append(valid[i:i+sequence_length,:])
valid_batch=np.array(valid_batch)
valid_batch=valid_batch.reshape(len(valid)-sequence_length+1,sequence_length,2)

x_valid = tf.convert_to_tensor(valid_batch[:,:,:])
y_train = x_train
y_valid = x_valid

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=0,mode='auto',baseline=None,restore_best_weights=False)
history = model.fit(x_train, y_train, batch_size=40, epochs = 1000, steps_per_epoch=3, validation_data=(x_valid,y_valid),verbose=1,callbacks=[early_stopping])
#history = model.fit(x_train, y_train, batch_size=20, epochs =50, steps_per_epoch=1, validation_data=(x_valid,y_valid),verbose=1)
plt.plot(history.history['loss'],label='train',c='b')
plt.plot(history.history['val_loss'],label='valid',c='orange')
plt.xlabel('epoch', fontsize='10')
plt.ylabel('loss', fontsize='10')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.show()

y_pred_train = model.predict(x_valid)
###########################SPE统计量的控制限
e=[]
for i in range(len(x_valid)):
    e.append(mean_squared_error(valid_batch[i,:,:],y_pred_train[i,:,:]))

px=kde.gaussian_kde(e,bw_method='silverman')###############得到概率密度估计函数
for limit_mse in np.arange(0,1,0.0001):
    if quad(lambda  x:px(x),-limit_mse,limit_mse)[0] > 0.99:
        limit_mse = limit_mse
        break######################对函数积分得到阈值

test2 = (data_test-mean)/std
plt.figure()      
plt.scatter(data_train[:,0],data_train[:,1],s=1,c='b',label='Training')
plt.scatter(data_fault[:,0],data_fault[:,1],s=1,c='g',label='Fault')
#plt.scatter(fault2_x,fault2_y,s=1,c='r',label='Fault 2')
plt.xlabel('x1',fontsize=12)
plt.ylabel('x2',fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()

test_batch = []
for i in range(len(test2)-sequence_length+1):
    test_batch.append(test2[i:i+sequence_length,:])
test_batch=np.array(test_batch)
test_batch=test_batch.reshape(len(test2)-sequence_length+1,sequence_length,2)

y_test = tf.convert_to_tensor(test_batch)
y_pred_test = model.predict(y_test)

fault2_h = []
for i in range(len(test_batch)):
    fault2_h.append(y_pred_test[i,-1,:])
f2_h = np.array(fault2_h).reshape(len(test_batch),2)

#
#####################SPE统计量监测
e2 = []
for i in range(len(test_batch)):
    e2.append(mean_squared_error(test_batch[i,:,:],y_pred_test[i,:,:]))
    
plt.figure()
plt.plot(e2,c='b')
plt.xlabel('Samples',fontsize=10)
plt.ylabel('SPE',fontsize=10)
plt.hlines(limit_mse,0,len(y_test),linewidth=3,color='r',label='99% SPE threshold')
plt.vlines(200-sequence_length+1,np.min(e2)-0.001,np.max(e2)-0.001,color='black',linestyle='--',linewidth=3)
plt.vlines(480-sequence_length+1,np.min(e2)-0.001,np.max(e2)-0.001,color='black',linestyle='--',linewidth=3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()

plt.figure()    
plt.plot(e,c='b')
plt.xlabel('Samples',fontsize=10)
plt.ylabel('SPE',fontsize=10)
plt.hlines(limit_mse,0,len(valid),linewidth=3,color='r',label='99% SPE threshold')
#plt.vlines(200-sequence_length+1,np.min(e2)-0.001,np.max(e2)+0.001,color='black',linestyle='--',linewidth=3)
#plt.vlines(480-sequence_length+1,np.min(e2)-0.001,np.max(e2)+0.001,color='black',linestyle='--',linewidth=3)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()

FAR = 0
for i in range(len(e2)):
    if i < 200-sequence_length+1 or i > 480-sequence_length+1:
        if e2[i] > limit_mse:
            FAR = FAR + 1
print('FAR = ' + str(FAR/(len(e2)-280)))

FDR = 0
for i in range(200-sequence_length+1,480-sequence_length+1):
    if e2[i] > limit_mse:
        FDR = FDR + 1
print('FDR = ' + str(FDR/280))
#model.save('sl = '+str(sequence_length)+'.h5')

        
x_pred_train = model.predict(x_valid)

train_h = []
for i in range(len(valid_batch)):
    train_h.append(x_pred_train[i,-1,:])
v_h = np.array(train_h).reshape(len(valid_batch),2)

train_r = v_h - valid[sequence_length-1:,:]
#f1_r = f1_h - test1[sequence_length-1:,:]
f2_r = f2_h[200-sequence_length+1:480-sequence_length+1,:] - test2[200:480,:]
def plot_cicle():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cir1 = Circle(xy = (0.0, 0.0), radius=np.sqrt(limit_mse*2), alpha=0.1,color='r',label='normal')
    ax.add_patch(cir1)

#    x, y = 0, 0
#    ax.plot(x, y, 'ro')

    plt.axis('scaled')
    plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
    plt.legend()
    plt.show()
 
#plot_cicle()
plt.figure()
a, b = np.sqrt(limit_mse), np.sqrt(limit_mse)
theta = np.arange(0, 2 * np.pi, np.pi / 100)
x = a * np.cos(theta)
y = b * np.sin(theta)
plt.plot(x, y,c='r',label='99% mse threshold',linestyle='--')
plt.xlabel('(x1-x1`)',fontsize=10)
plt.ylabel('(x2-x2`)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()

plt.scatter(train_r[:,0],train_r[:,1],s=1,c='b',label='Training')
#plt.scatter(f1_r[:,0],f1_r[:,1],s=1,c='g',label='Fault1')
plt.scatter(f2_r[:,0],f2_r[:,1],s=1,c='r',label='Fault')
plt.xlim(-0.1,0.25)
plt.ylim(-0.1,0.25)
plt.legend()

layer_name=[]
for layer in model.layers:
    layer_name.append(layer.name)

#tf.config.experimental_run_functions_eagerly(False)   
model_1 = model.get_layer(layer_name[0])
train_latent = model_1(tf.cast(x_train,tf.float32)).numpy().reshape(len(x_train),output_size2)
test1_latent = model_1(tf.cast(y_test1,tf.float32)).numpy().reshape(len(y_test1),output_size2)
test2_latent = model_1(tf.cast(y_test2,tf.float32)).numpy().reshape(len(y_test2),output_size2)

plt.figure()
plt.scatter(train_latent[:,0],train_latent[:,1],s=1,c='b',label='Training')
plt.scatter(test1_latent[:,0],test1_latent[:,1],s=1,c='g',label='Fault1')
plt.scatter(test2_latent[:,0],test2_latent[:,1],s=1,c='r',label='Fault2')
plt.xlabel('PC1',fontsize=10)
plt.xlabel('PC2',fontsize=10)
#plt.xlim(-0.6,0.6)
#plt.ylim(-0.6,0.6)
plt.legend()
