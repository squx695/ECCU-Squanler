import time
import numpy as np
import pandas as pd
import schedule
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
import Config
import math
import os
import math

class Squanler:
    def __init__(self, config: Config):
        self.config = config
        self.k8s_util = KubernetesClient(config)
        self.prom_util = PrometheusClient(config)
        self.scale_interval = config.scale_interval

        self.EAOL = 2.6

    def scale(self):
        
        ECCU = self.prom_util.get_ECCU(self.EAOL)
        STRATEGY = [0 for svc in self.config.svcs]
        
        for interface, ECCU in ECCU.items():
            ISH_i = self.config.interfaces[interface][0]
            Target = self.config.interfaces[interface][1]

            _RPS = ECCU / (self.EAOL + Target / 1000)

            for j in range(len(ISH_i)):
                STRATEGY[j] += ISH_i[j] * _RPS

        for i in range(len(self.config.svcs)):
            count = math.ceil(STRATEGY[i] / self.config.request)
            self.k8s_util.patch_scale(self.config.svcs[i], count)
    
    def start(self):
        print("Squanler goes...xwx...!")
        
        self.scale()
        schedule.every(self.scale_interval).seconds.do(self.scale)

        time_start = time.time()
        while True:
            time_c = time.time() - time_start
            if time_c > self.config.duration:
                schedule.clear()
                break
            schedule.run_pending()
            
        print("Squanler stops...xwx...!")
