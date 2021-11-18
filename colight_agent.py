import numpy as np 
import os 
import pickle  
from agent import Agent
import random 
import time
"""
Model for CoLight in paper "CoLight: Learning Network-level Cooperation for Traffic Signal
Control", in submission.
"""
import keras
from keras import backend as K
from keras.optimizers import RMSprop
import tensorflow as tf
# from keras.layers import Dense, Input, Lambda, Reshape 
from keras.models import Model # , model_from_json, load_model
# from keras.utils import to_categorical
# from keras.engine.topology import Layer
# from keras.callbacks import EarlyStopping, TensorBoard

# pytorch-port
from pathlib import Path
# from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim  import Adam

# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)


# class RepeatVector3D(Layer):
#     def __init__(self,times,**kwargs):
#         super(RepeatVector3D, self).__init__(**kwargs)
#         self.times = times
# 
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.times, input_shape[1],input_shape[2])
# 
#     def call(self, inputs):
#         #[batch,agent,dim]->[batch,1,agent,dim]
#         #[batch,1,agent,dim]->[batch,agent,agent,dim]
# 
#         return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])
# 
# 
#     def get_config(self):
#         config = {'times': self.times}
#         base_config = super(RepeatVector3D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def to_categorical(n_classes, x):
    return np.eye(n_classes)[x]

class CoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None,
                 dic_path=None, cnt_round=None, best_round=None,
                 bar_round=None,intersection_id="0"):
        """
            #1. compute the (dynamic) static Adjacency matrix, compute for each state
            -2. #neighbors: 5 (1 itself + W,E,S,N directions)
            -3. compute len_features
            -4. self.num_actions
        """
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.att_regulatization=dic_agent_conf['att_regularization']
        self.CNN_layers=dic_agent_conf['CNN_layers']

        #TODO: n_agents should pass as parameter
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        self.vec = np.zeros((1,self.num_neighbors))
        self.vec[0][0] = 1

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature =self.compute_len_feature()

        # Resume previous procedure
        # if cnt_round == 0:
            # initialization
        self.q_network = self.build_network()
        self.optimizer = Adam(self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        self.q_network_bar = self.build_network()

        if cnt_round == 0 and os.listdir(self.dic_path["PATH_TO_MODEL"]):
            file_name = f'round_0_inter_{intersection_id}.h5'
            self.load_network(file_name)
        self.q_network_bar.load_state_dict(self.q_network.state_dict())

        if cnt_round > 0:
            try:
                if best_round:
                    # use model pool
                    file_name = "round_{0}_inter_{1}".format(best_round, self.intersection_id)
                    self.load_network(file_name)

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        # load q_bar network from model pool
                        file_name = "round_{0}_inter_{1}".format(bar_round, self.intersection_id)
                        self.load_network_bar(file_name)
                    else:
                        update_q_bar = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
                        if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                                bar_round = max((best_round - 1) //  update_q_bar * update_q_bar, 0)
                            else:
                                bar_round = max(best_round - update_q_bar, 0)
                        else:
                            bar_round = max(best_round - update_q_bar, 0)
                        file_name = "round_{0}_inter_{1}".format(bar_round, self.intersection_id)
                        self.load_network_bar(file_name)
                else:
                    # not use model pool
                    #TODO how to load network for multiple intersections?
                    # print('init q load')
                    self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                    # print('init q_bar load')

                    update_q_bar = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:

                            bar_round = max((cnt_round - 1) //  update_q_bar * update_q_bar, 0)
                        else:
                            bar_round = max(cnt_round - update_q_bar, 0)
                    else:
                        bar_round = max(cnt_round - update_q_bar, 0)

                    file_name = "round_{0}_inter_{1}".format(bar_round, self.intersection_id)
                    self.load_network_bar(file_name)
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def compute_len_feature(self):
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            elif feature_name=="lane_num_vehicle":
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    # TODO: pytorch-port
    # def MLP(self,In_0,layers=[128,128]):
    #     """
    #     Currently, the MLP layer 
    #     -input: [batch,#agents,feature_dim]
    #     -outpout: [batch,#agents,128]
    #     """
    #     # In_0 = Input(shape=[self.num_agents,self.len_feature])
    #     for layer_index,layer_size in enumerate(layers):
    #         if layer_index==0:
    #             h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
    #         else:
    #             h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

    #     return h





    # def MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
    #     """
    #     input:[bacth,agent,128]
    #     output:
    #     -hidden state: [batch,agent,32]
    #     -attention: [batch,agent,neighbor]
    #     """
    #     """
    #     agent repr
    #     """
    #     print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
    #     #[batch,agent,dim]->[batch,agent,1,dim]
    #     agent_repr=Reshape((self.num_agents,1,d))(In_agent)

    #     """
    #     neighbor repr
    #     """
    #     #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
    #     neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
    #     print("neighbor_repr.shape", neighbor_repr.shape)
    #     #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
    #     neighbor_repr=Lambda(lambda x:K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])
    #     print("neighbor_repr.shape", neighbor_repr.shape)
    #     """
    #     attention computation
    #     """
    #     #multi-head
    #     #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
    #     agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
    #     #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
    #     agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
    #     agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
    #     #agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
    #     #[batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

    #     neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
    #     #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
    #     print("DEBUG",neighbor_repr_head.shape)
    #     print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors,dv,nv)
    #     neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_repr_head)
    #     neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
    #     #neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)        
    #     #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
    #     att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
    #     #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
    #     att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)


    #     #self embedding again
    #     neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
    #     neighbor_hidden_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_hidden_repr_head)
    #     neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head)
    #     out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head])
    #     out=Reshape((self.num_agents,dv))(out)
    #     out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
    #     return out,att_record





    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        # l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        l = to_categorical(self.num_agents, adjacency_index_new)
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        if bar:
            # all_output= self.q_network_bar.predict([total_features,total_adjs])
            all_output= self.q_network_bar(torch.tensor(total_features), torch.tensor(total_adjs))
        else:
            # all_output= self.q_network.predict([total_features,total_adjs])
            all_output = self.q_network(torch.tensor(total_features), torch.tensor(total_adjs))
        action, attention = all_output[0].clone().detach().cpu().numpy(), all_output[1].clone().detach().cpu().numpy()

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action,axis=-1),axis=-1)
        random_action=np.reshape(np.random.randint(self.num_actions,size=1*self.num_agents),(1,self.num_agents,1))
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act, attention


    def choose_action(self, count, state):
        '''
        choose the best action for current state
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state])
        return act[0],attention[0]


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """ """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action=[]
        _reward=[]

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]    
        _features,_adjs,q_values,_=self.action_att_predict(_state)   
        _next_features,_next_adjs,_,attention= self.action_att_predict(_next_state)
        #target_q_values:[batch,agent,action]
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file
    def build_network( self, MLP_layers=[32,32], Output_layers=[]):

        CNN_layers = self.CNN_layers

        CNN_heads = [1]*len(CNN_layers)

        """
        layer definition
        """
        start_time=time.time()
        # WARNING: Those options were turned down
        assert not any(Output_layers)
        assert len(CNN_layers) == 1
        assert not self.att_regulatization
        assert CNN_heads == [1] * len(CNN_layers)
        assert len(CNN_layers)==len(CNN_heads)

        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """

        # In=list()
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors,agents]

        # In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        # In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix"))
        n_feature = self.len_feature
        n_embeddings = MLP_layers
        n_agents = self.num_agents
        n_neighbors = self.num_neighbors
        n_input = n_embeddings[-1]
        n_hidden, n_output = CNN_layers[0]
        n_actions = self.num_actions
        n_concat = CNN_heads[0] # Should be 1
        model = GraphAttention(
            n_feature, n_embeddings, n_actions, n_agents,
            n_neighbors=n_neighbors, n_input=n_input, n_hidden=n_hidden,
            n_output=n_output, n_concat=n_concat, suffix=0
        )

        # """
        # Currently, the MLP layer
        # -input: [n_batch,n_agent,n_feature_dim]
        # -outpout: [#n_agent,n_batch,128]
        # """
        # feature=self.MLP(In[0],MLP_layers)

        # Embedding_end_time=time.time()


        # #TODO: remove the dense setting
        # #feature:[batch,agents,feature_dim]
        # att_record_all_layers=list()
        # print("CNN_heads:", CNN_heads)
        # for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
        #     print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
        #     if CNN_layer_index==0:
        #         h,att_record=self.MultiHeadsAttModel(
        #             feature,
        #             In[1],
        #             l=self.num_neighbors,
        #             d=MLP_layers[-1],
        #             dv=CNN_layer_size[0],
        #             dout=CNN_layer_size[1],
        #             nv=CNN_heads[CNN_layer_index],
        #             suffix=CNN_layer_index
        #             )
        #     else:
        #         h,att_record=self.MultiHeadsAttModel(
        #             h,
        #             In[1],
        #             l=self.num_neighbors,
        #             d=MLP_layers[-1],
        #             dv=CNN_layer_size[0],
        #             dout=CNN_layer_size[1],
        #             nv=CNN_heads[CNN_layer_index],
        #             suffix=CNN_layer_index
        #             )
        #     att_record_all_layers.append(att_record)

        # #action prediction layer
        # #[batch,agent,32]->[batch,agent,action]
        # out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer')(h)
        # #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        # model=Model(inputs=In,outputs=[out,att_record_all_layers])

        # model.compile(
        #     optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
        #     loss=self.dic_agent_conf["LOSS_FUNCTION"],
        #     loss_weights=[1,0])
        # model.summary()

        print(model)
        # network_end=time.time()

        # print('build_Input_end_time：',Input_end_time-start_time)
        # print('embedding_time:',Embedding_end_time-Input_end_time)
        # print('total time:',network_end-start_time)
        return model

    def train_network(self, dic_exp_conf):
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        # early_stopping = EarlyStopping(
        #     monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')


        # original code sets loss_weights=0
        # TODO: set evaluation
        targets = torch.tensor(self.Y_total[0], requires_grad=False)
        x, y = torch.tensor(self.Xs[0]), torch.tensor(self.Xs[1])
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs, _= self.q_network(x, y)
            loss = nn.MSELoss()(outputs, targets)
            print(f'{epoch}:\tloss\t{loss.clone().detach().cpu()}')
            loss.backward()
            self.optimizer.step()
        # hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3,
        #                           callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])


    # pytorch-port
    # def build_network_from_copy(self, network_copy):

    #     '''Initialize a Q network from a copy'''
    #     network_structure = network_copy.to_json()
    #     network_weights = network_copy.get_weights()
    #     network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
    #     network.set_weights(network_weights)

    #     if self.att_regulatization:
    #         network.compile(
    #             optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
    #             loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
    #             loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
    #     else:
    #         network.compile(
    #             optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
    #             loss=self.dic_agent_conf["LOSS_FUNCTION"],
    #             loss_weights=[1,0])

    #     return network

    def load_network(self, file_name, file_path=None):
        file_path = self.save_path if file_path is None else Path(file_path)

        state_dict = torch.load((file_path / f'{file_name}.h5').as_posix())
        self.q_network.load_state_dict(state_dict, strict=False)
        # self.q_network = load_model(
        #     os.path.join(file_path, "%s.h5" % file_name),
        #     custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        file_path = self.save_path if file_path is None else Path(file_path)

        state_dict = torch.load((file_path / f'{file_name}.h5').as_posix())
        self.q_network_bar.load_state_dict(state_dict, strict=False)
        # self.q_network_bar = load_model(
        #     os.path.join(file_path, "%s.h5" % file_name),
        #     custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name)

    def save_network(self, file_name):
        file_path = (self.save_path / f'{file_name}.h5')
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.q_network.state_dict(), file_path.as_posix())

    def save_network_bar(self, file_name):
        file_path = (self.save_path / f'{file_name}.h5')
        file_path.parent.mkdir(exist_ok=True)
        torch.save(self.q_network_bar.state_dict(), file_path.as_posix())

    @property
    def save_path(self):
        return Path(self.dic_path["PATH_TO_MODEL"])

    # def save_network(self, file_name):
    #     self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    # def save_network_bar(self, file_name):
    #     self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

class GraphAttention(nn.Module):
    def __init__(self, n_feature, n_embeddings, n_actions, n_agents,
                 n_neighbors=5, n_input=128, n_hidden=16, n_output=128,
                 n_concat=1, suffix=-1):
        super(GraphAttention, self).__init__()

        self.embeddings = MLP(n_feature, n_embeddings)
        self.attention = GraphAttentionLayer(
            n_agents, n_neighbors, n_input=n_embeddings[-1], n_hidden=n_hidden,
            n_output=n_output, n_concat=n_concat, suffix=suffix
        )
        self.prediction = nn.Linear(n_output, n_actions)

        self.add_module('embeddings', self.embeddings)
        self.add_module('attention', self.attention)
        self.add_module('prediction', self.prediction)

    def forward(self, x, y):
        x = self.embeddings(x.float())

        x, y = self.attention(x, y.float())

        x = self.prediction(x)

        return x, y

class MLP(nn.Module):
    def __init__(self, n_feature, n_hiddens):
        super(MLP, self).__init__()
        self.fcs = []
        self.n_feature = n_feature
        self.n_hiddens = n_hiddens
        n_in = n_feature
        for n_layer, n_out in enumerate(n_hiddens):
            self.fcs.append(nn.Linear(n_in, n_out))
            self.add_module(f'fc_{n_layer}', self.fcs[-1])
            n_in = n_out

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)
        return x

class GraphAttentionLayer(nn.Module):

    def __init__(self, n_agents, n_neighbors, n_input=128, n_hidden=16, n_output=128, n_concat=1,suffix=-1):
        '''
        input:[bacth,agent,128]
        output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
        print("In_agent.shape,In_neighbor.shape,n_neighbors, n_input, n_hidden, n_output, n_concat", In_agent.shape,In_neighbor.shape,n_neighbors, n_input, n_hidden, n_output, n_concat)
        In_agent.shape,In_neighbor.shape,n_neighbors, n_input, n_hidden, n_output, n_concat (?, 36, 32) (?, 36, 5, 36) 5 32 32 32 1
        '''
        super(GraphAttentionLayer, self).__init__()
        # Dimension parameters
        self.n_agents = n_agents
        self.n_neighbors = n_neighbors
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_concat = n_concat
        self.n_output = n_output

        # Fully connected layers
        self.fc_1 = nn.Linear(n_input, n_hidden * n_concat)
        self.fc_2 = nn.Linear(n_hidden * n_concat, n_hidden * n_concat)
        self.fc_3 = nn.Linear(n_hidden * n_concat, n_hidden * n_concat)
        self.fc_4 = nn.Linear(n_hidden * n_concat, n_output)

        self.add_module(f'agent_{suffix}', self.fc_1)
        self.add_module(f'neighbor_{suffix}', self.fc_2)
        self.add_module(f'inner_{suffix}', self.fc_3)
        self.add_module(f'output_{suffix}', self.fc_4)

        self._agent_head_shape = (-1, n_agents, 1, n_concat, n_hidden)
        self._neighbor_head_shape = (-1, n_agents, 1, n_neighbors, n_hidden)
        self._att_shape = (-1, n_agents, n_concat, n_neighbors)
        self._hidden_shape = (-1, n_agents, n_concat, n_neighbors, n_hidden)
        self._output_shape = (-1, n_agents, n_hidden)


    def forward(self, x, y):
        '''
        Parameters:
        ------------
        * x: [n_batch, n_agent, n_hidden]
        * y: [n_batch, n_agents, n_neighbors, n_agents]

        Returns:
        --------
        * attention: [batch,agent,neighbor]
        * hidden state: [batch,agent,32]
        '''

        '''
                    1. REPRESENTATIONS
        '''
        # 1.1 agent repr
        #[batch,agent,dim]->[batch,agent,1,dim]

        agent_repr = x.unsqueeze(2)

        # 1.2 neighbor repr
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr = x.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        # print("neighbor_repr.shape", neighbor_repr.shape)

        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        z = torch.transpose(y, dim0=-2, dim1=-1)
        neighbor_repr = torch.einsum('ijmn, ijmk -> ijnk', z, neighbor_repr)
        # [n_batch, n_agents, n_neighbor, n_hidden]
        # print("neighbor_repr.shape", neighbor_repr.shape)

        '''
                    2. ATTENTION
        '''
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1, n_hidden * n_concat]
        agent_repr_head = F.relu(self.fc_1(agent_repr))


        # TODO: pytorch-port
        #[n_batch, n_agents, 1, n_hidden] -> [n_batch, n_agent, n_concat, 1, n_hidden]
        #[batch, agent, neighbor, dim]->[batch,agent,neighbor,n_hidden*n_concat]
        agent_repr_head = agent_repr_head.view(self._agent_head_shape)

        # TODO: pytorch-port
        #[batch,agent,neighbor,n_hidden,n_concan_concat]->[batch,agent,nv,neighbor,n_hidden]
        neighbor_repr_head = F.relu(self.fc_2(neighbor_repr))

        # print("DEBUG",neighbor_repr_head.shape)
        # print("self.num_agents,self.num_neighbors,n_hidden,n_concat", self.n_agents,self.n_neighbors, self.n_hidden, self.n_concat)
        neighbor_repr_head = neighbor_repr_head.view(self._neighbor_head_shape)
        #[batch,agent,n_concat,1,n_hidden]x[batch,agent,n_concat,neighbor,n_hidden]->[batch,agent,n_concat,1,neighbor]
        scores = torch.einsum('ijklm, ijknm -> ijkln', agent_repr_head, neighbor_repr_head)
        att = F.softmax(scores, dim=-1)
        #[batch,agent,n_concat,1,neighbor]->[batch,agent,n_concat,neighbor]
        att_record = att.view(self._att_shape)

        '''
                    3. OUTPUT
        '''

        neighbor_hidden_repr_head = self.fc_3(neighbor_repr)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.view(self._hidden_shape)
        z = torch.transpose(att, dim0=-2, dim1=-1)
        scores = torch.einsum('ijklm, ijkln -> ijkmn', z, neighbor_hidden_repr_head)
        output = self.fc_4(torch.mean(scores, dim=2)).view(self._output_shape)

        return output, att_record
