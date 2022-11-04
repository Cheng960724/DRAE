class MIM_AE(tf.keras.layers.Layer):
    def __init__(self,output_size1,output_size_N, output_size_S, output_size2, return_sequences,**kwargs):
#        tf.set_random_seed(1)
        super(MIM_AE,self).__init__()
        self.output_size1 = output_size1#############################ht
        self.output_size_N = output_size_N#############################dt
        self.output_size_S = output_size_S#############################tt
        self.output_size2 = output_size2#############################Ht        
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        super(MIM_AE,self).build(input_shape)
        input_size = int(input_shape[-1])
        
        self.wf = self.add_weight('wf', shape=(input_size,output_size1))
        self.wi = self.add_weight('wi', shape=(input_size,output_size1))
        self.wo = self.add_weight('wo', shape=(input_size,output_size1))
        self.wc = self.add_weight('wc', shape=(input_size,output_size1))
        
        self.uf = self.add_weight('uf', shape=(output_size1,output_size1))
        self.ui = self.add_weight('ui', shape=(output_size1,output_size1))
        self.uo = self.add_weight('uo', shape=(output_size1,output_size1))
        self.uc = self.add_weight('uc', shape=(output_size1,output_size1))
        
        self.bf = self.add_weight('bf', shape=(1,output_size1))
        self.bi = self.add_weight('bi', shape=(1,output_size1))
        self.bo = self.add_weight('bo', shape=(1,output_size1))
        self.bc = self.add_weight('bc', shape=(1,output_size1))
        
        self.wf1 = self.add_weight('wf1', shape=(output_size1,output_size_N))
        self.wi1 = self.add_weight('wi1', shape=(output_size1,output_size_N))
        self.wo1 = self.add_weight('wo1', shape=(output_size1,output_size_N))
        self.wc1 = self.add_weight('wc1', shape=(output_size1,output_size_N))
        
        self.uf1 = self.add_weight('uf1', shape=(output_size_N,output_size_N))
        self.ui1 = self.add_weight('ui1', shape=(output_size_N,output_size_N))
        self.uo1 = self.add_weight('uo1', shape=(output_size_N,output_size_N))
        self.uc1 = self.add_weight('uc1', shape=(output_size_N,output_size_N))
        
        self.bf1 = self.add_weight('bf1', shape=(1,output_size_N))
        self.bi1 = self.add_weight('bi1', shape=(1,output_size_N))
        self.bo1 = self.add_weight('bo1', shape=(1,output_size_N))
        self.bc1 = self.add_weight('bc1', shape=(1,output_size_N))
        
        self.wf2 = self.add_weight('wf2', shape=(output_size_N,output_size_S))
        self.wi2 = self.add_weight('wi2', shape=(output_size_N,output_size_S))
        self.wo2 = self.add_weight('wo2', shape=(output_size_N,output_size_S))
        self.wc2 = self.add_weight('wc2', shape=(output_size_N,output_size_S))
        
        self.uf2 = self.add_weight('uf2', shape=(output_size2,output_size_S))
        self.ui2 = self.add_weight('ui2', shape=(output_size2,output_size_S))
        self.uo2 = self.add_weight('uo2', shape=(output_size2,output_size_S))
        self.uc2 = self.add_weight('uc2', shape=(output_size2,output_size_S))
        
        self.bf2 = self.add_weight('bf2', shape=(1,output_size_S))
        self.bi2 = self.add_weight('bi2', shape=(1,output_size_S))
        self.bo2 = self.add_weight('bo2', shape=(1,output_size_S))
        self.bc2 = self.add_weight('bc2', shape=(1,output_size_S))
        
        self.vo = self.add_weight('vo', shape=(output_size_S,output_size_S))
        
        self.wi3 = self.add_weight('wi3', shape=(output_size1,output_size2))
        self.wo3 = self.add_weight('wo3', shape=(output_size1,output_size2))
        self.wc3 = self.add_weight('wc3', shape=(output_size1,output_size2))
        
        self.ui3 = self.add_weight('ui3', shape=(output_size2,output_size2))
        self.uo3 = self.add_weight('uo3', shape=(output_size2,output_size2))
        self.uc3 = self.add_weight('uc3', shape=(output_size2,output_size2))
        
        self.bi3 = self.add_weight('bi3', shape=(1,output_size2))
        self.bo3 = self.add_weight('bo3', shape=(1,output_size2))
        self.bc3 = self.add_weight('bc3', shape=(1,output_size2))
        
    
    def call(self, x):
        sequence_outputs1 = []
        sequence_outputs_N = []
        sequence_outputs_S = []
        sequence_outputs2 = []
        for i in range(sequence_length):
            if i == 0:
                xt = x[:, 0, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + self.bo)
                gt = tf.tanh(tf.matmul(xt, self.wc) + self.bc)
                ct = it * gt
                ht = ot* tf.tanh(ct)
                
                ft1 = tf.sigmoid(tf.matmul(ht, self.wf1) + self.bf1)
                it1 = tf.sigmoid(tf.matmul(ht, self.wi1) + self.bi1)
                ot1 = tf.sigmoid(tf.matmul(ht, self.wo1) + self.bo1)
                gt1 = tf.tanh(tf.matmul(ht, self.wc1) + self.bc1)
                nt =  it1 * gt1
                dt = ot1* tf.tanh(nt)
                
                ft2 = tf.sigmoid(tf.matmul(dt, self.wf2) + self.bf2)
                it2 = tf.sigmoid(tf.matmul(dt, self.wi2) + self.bi2)
                gt2 = tf.tanh(tf.matmul(dt, self.wc2) + self.bc2)
                st = it2 * gt2
                ot2 = tf.sigmoid(tf.matmul(dt, self.wo2) + tf.matmul(st, self.vo) + self.bo2)
                tt = ot2* tf.tanh(st)
                
                it3 = tf.sigmoid(tf.matmul(ht, self.wi3) + self.bi3)
                gt3 = tf.tanh(tf.matmul(ht, self.wc3) + self.bc3)
                Ct = tt + it3 * gt3
                ot3 = tf.sigmoid(tf.matmul(ht, self.wo3) + self.bo3)
                Ht = ot3* tf.tanh(Ct)
                
            else:
                xt = x[:, i, :]
                ft = tf.sigmoid(tf.matmul(xt, self.wf) + tf.matmul(ht, self.uf) + self.bf)
                it = tf.sigmoid(tf.matmul(xt, self.wi) + tf.matmul(ht, self.ui) + self.bi)
                ot = tf.sigmoid(tf.matmul(xt, self.wo) + tf.matmul(ht, self.uo) + self.bo)
                gt = tf.tanh(tf.matmul(xt, self.wc) + tf.matmul(ht, self.uc) + self.bc)
                ct = ft * ct + it * gt
                ht1 = ot* tf.tanh(ct)############################################# ht1为第t个时刻的ht，计算后再将ht1赋值给ht
                hht = tf.subtract(ht1,ht)#########################################hht定义为差分项
                
                ft1 = tf.sigmoid(tf.matmul(hht, self.wf1) + tf.matmul(nt, self.uf1) + self.bf1)
                it1 = tf.sigmoid(tf.matmul(hht, self.wi1) + tf.matmul(nt, self.ui1) + self.bi1)
                gt1 = tf.tanh(tf.matmul(hht, self.wc1) + tf.matmul(nt, self.uc1) + self.bc1)
                nt =  ft1 * nt + it1 * gt1
                ot1 = tf.sigmoid(tf.matmul(hht, self.wo1) + tf.matmul(nt, self.uo1)+ self.bo1)
                dt = ot1* tf.tanh(nt)
                
                ft2 = tf.sigmoid(tf.matmul(dt, self.wf2) + tf.matmul(Ct, self.uf2) + self.bf2)
                it2 = tf.sigmoid(tf.matmul(dt, self.wi2) + tf.matmul(Ct, self.ui2) + self.bi2)
                gt2 = tf.tanh(tf.matmul(dt, self.wc2) +  tf.matmul(Ct, self.uc2) + self.bc2)
                st = ft2 * st + it2 * gt2
                ot2 = tf.sigmoid(tf.matmul(dt, self.wo2) + tf.matmul(Ct, self.uo2) + tf.matmul(st, self.vo) + self.bo2)
                tt = ot2* tf.tanh(st)
                
                it3 = tf.sigmoid(tf.matmul(ht, self.wi3) + tf.matmul(Ht, self.ui3) + self.bi3)
                gt3 = tf.tanh(tf.matmul(ht, self.wc3) + tf.matmul(Ht, self.uc3) + self.bc3)
                Ct = tt + it3 * gt3
                ot3 = tf.sigmoid(tf.matmul(ht, self.wo3) + tf.matmul(Ht, self.uo3) + self.bo3)
                Ht = ot3* tf.tanh(Ct)
                ht = ht1
            sequence_outputs1.append(ht)
            sequence_outputs_N.append(dt)
            sequence_outputs_S.append(tt)
            sequence_outputs2.append(Ht)
        sequence_outputs1 = tf.stack(sequence_outputs1)
        sequence_outputs1 = tf.transpose(sequence_outputs1, (1, 0, 2))
#        
        sequence_outputs_N = tf.stack(sequence_outputs_N)
        sequence_outputs_N = tf.transpose(sequence_outputs_N, (1, 0, 2))
        
        sequence_outputs_S = tf.stack(sequence_outputs_S)
        sequence_outputs_S = tf.transpose(sequence_outputs_S, (1, 0, 2))
        
        sequence_outputs2 = tf.stack(sequence_outputs2)
        sequence_outputs2 = tf.transpose(sequence_outputs2, (1, 0, 2))
        if self.return_sequences:
            return sequence_outputs2
        return sequence_outputs2[:, -1, :]
    def get_config(self):
        
        
        config = {'output_size1':self.output_size1,
                       'output_size_N':self.output_size_N,
                       'output_size_S':self.output_size_S,
                       'output_size2':self.output_size2,
                       'return_sequences':self.return_sequences}
        base_config = super(MIM_AE,self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
