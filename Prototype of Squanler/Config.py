import time

def getNowTime():
    return int(round(time.time()))

class Config:
    def __init__(self):
        
        # for now, we only consider these three status codes
        self.cycles = ['2..', '302', '5..']
        
        self.namespace = 'hipster'
        self.ingress_namespace = 'istio-system'
        self.ingress = 'istio-ingressgateway'
        
        self.redict_chain = [
            ['/setCurrency && POST', '/ && GET'],
        ]

        self.SLO = 200
        self.scale_interval = 30
        self.request = 150

        self.svcs = ['frontend','adservice','cartservice','checkoutservice','currencyservice','emailservice','paymentservice','productcatalogservice','recommendationservice','shippingservice']
        
        # your ISH
        self.interfaces = {
            "/ && GET": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], self.SLO],
            "/product/{id} && GET": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], self.SLO],
            '/cart && GET': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], self.SLO],
            '/cart/checkout && POST': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], self.SLO],
            '/setCurrency && POST': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], self.SLO],
        }

        self.max_pod = 100
        self.min_pod = 1

        self.k8s_config = 'admin.conf'

        self.duration = 1 * 60 * 60
        self.start = getNowTime()
        self.end = self.start + self.duration
        self.step = 5

        self.prom_range_url = "http://XXX.XXX.XX.XXX:XXXXX//api/v1/query_range"
        self.prom_no_range_url = "http://XXX.XXX.XX.XXX:XXXXX//api/v1/query"