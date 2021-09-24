import os
import copy
import time
import sys
from multiprocessing import Process, Pool

from anon_env import AnonEnv
from config import DIC_AGENTS
from updater import load_sample_for_agents, update_network_for_agents
from construct_sample import ConstructSample

class Trainer:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']

        # Builds seed.
        self.seed = cnt_gen * dic_exp_conf['SEED_GROWTH_FACTOR'] \
             + dic_exp_conf['SEED_BASE']
        self.dic_exp_conf['SEED'] = self.seed
        self.dic_agent_conf['SEED'] = self.seed
        self.dic_traffic_env_conf['SEED'] = self.seed

        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = AnonEnv(
              path_to_log = self.path_to_log,
              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
              dic_traffic_env_conf = self.dic_traffic_env_conf,
        )
        self.env.reset()
        self.dic_traffic_env_conf['traffic_light_phases'] = self.env.traffic_light_phases

        start_time = time.time()

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            # FIXME: 'True' phase
            intersec = self.env.list_intersection[i]
            #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
            if agent_name=='CoLight_Signal':
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    best_round=best_round,
                    intersection=intersec,
                    intersection_id=str(i)
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    best_round=best_round,
                    intersection=intersec,
                    intersection_id=str(i)
                )
            self.agents[i] = agent

        # Load pre-pretrained model for previous episode. 
        if self.cnt_round > 0:
            import ipdb; ipdb.set_trace()
            load_sample_for_agents(
                self.dic_agent_conf,
                self.dic_exp_conf,
                self.dic_traffic_env_conf,
                self.dic_path,
                self.agents
            )
            update_network_for_agents(
                self.dic_exp_conf,
                self.dic_traffic_env_conf,
                self.dic_path,
                self.agents,
                self.cnt_round
            )
        print("Create intersection agent time: ", time.time()-start_time)

    def train(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        step_max = \
            int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"])
        if step_max % self.dic_exp_conf["NUM_TRAIN_UPDATES"] != 0:
            ValueError('Parameter NUM_TRAIN_UPDATES must divide RUN_COUNTS/MIN_ACTION_TIME')
        train_interval = int(step_max / self.dic_exp_conf["NUM_TRAIN_UPDATES"])
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        while True:
            action_list = []
            step_start_time = time.time()

            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                if self.dic_exp_conf["MODEL_NAME"] in ["CoLight","GCN", "SimpleDQNOne"]:
                    one_state = state
                    if self.dic_exp_conf["MODEL_NAME"] == 'CoLight':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
                        action = self.agents[i].choose_action(step_num, one_state)
                    else: # simpleDQNOne
                        if True:
                            action = self.agents[i].choose_action(step_num, one_state)
                        else:
                            action = self.agents[i].choose_action_separate(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)


            next_state, reward, done, _ = self.env.step(action_list)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                        time.time()-step_start_time))
            state = next_state
            step_num += 1
            print(step_num)

            # Every refresh period.
            if step_num % train_interval == 0:
                print(f'refresh :{int(step_num / train_interval)}, step: {step_num}')
                self.env.batch_log(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])

                self.make_samples()
                load_sample_for_agents(
                    self.dic_agent_conf,
                    self.dic_exp_conf,
                    self.dic_traffic_env_conf,
                    self.dic_path,
                    self.agents
                )
                update_network_for_agents(
                    self.dic_exp_conf,
                    self.dic_traffic_env_conf,
                    self.dic_path,
                    self.agents,
                    self.cnt_round
                )
            if done or step_num == step_max: break
        running_time = time.time() - running_start_time
        log_start_time = time.time()

        print("start logging")
        self.env.batch_log(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
        log_time = time.time() - log_start_time
        self.env.info_log()
        self.env.emission_log()

        self.env.end_sumo()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
        return

    def make_samples(self):
        print("==============  make samples =============")
        # make samples and determine which samples are good
        making_samples_start_time = time.time()

        train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
        if not os.path.exists(train_round):
            os.makedirs(train_round)
        cs = ConstructSample(path_to_samples=train_round, cnt_round=self.cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        cs.make_reward_for_system()
        # EvaluateSample()
        making_samples_end_time = time.time()
        making_samples_total_time = making_samples_end_time - making_samples_start_time
